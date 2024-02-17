import torch
from torch import nn
from torch.nn import functional as F
from model.embedder import get_embedder


class InvLoss(nn.Module):
    def __init__(self, idr_rgb_weight, eikonal_weight, mask_weight, normal_l1_weight, normal_cos_weight, alpha,
                    sg_rgb_weight, kl_weight, latent_smooth_weight, albedo_weight, albedo_grad_weight,
                    brdf_multires=10, loss_type='L1'):
        super().__init__()
        self.idr_rgb_weight = idr_rgb_weight
        self.eikonal_weight = eikonal_weight
        self.mask_weight = mask_weight
        self.normal_l1_weight = normal_l1_weight
        self.normal_cos_weight = normal_cos_weight
        self.alpha = alpha

        self.sg_rgb_weight = sg_rgb_weight
        self.kl_weight = kl_weight
        self.latent_smooth_weight = latent_smooth_weight
        self.albedo_weight = albedo_weight
        self.albedo_grad_weight = albedo_grad_weight
        self.brdf_multires = brdf_multires
        
        if loss_type == 'L1':
            print('Using L1 loss for comparing images!')
            self.img_loss = nn.L1Loss(reduction='sum')
        elif loss_type == 'L2':
            print('Using L2 loss for comparing images!')
            self.img_loss = nn.MSELoss(reduction='sum')
        else:
            raise Exception('Unknown loss_type!')
        
        self.albedo_criterion = nn.L1Loss(reduction='sum')
        
    
    def get_rgb_loss(self,rgb_values, rgb_gt, network_object_mask, object_mask):
        if (network_object_mask & object_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float()

        rgb_values = rgb_values[network_object_mask & object_mask]
        rgb_gt = rgb_gt.reshape(-1, 3)[network_object_mask & object_mask]
        rgb_loss = self.img_loss(rgb_values, rgb_gt) / float(object_mask.shape[0])
        return rgb_loss
    
    # albedo loss
    def get_albedo_loss(self, albedo_values, albedo_gt, network_object_mask, object_mask):
        if (network_object_mask & object_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float()
        albedo_values = albedo_values[network_object_mask & object_mask]
        albedo_gt = albedo_gt.reshape(-1, 3)[network_object_mask & object_mask]
        albedo_loss = self.albedo_criterion(albedo_values, albedo_gt) / float(object_mask.shape[0])
        return albedo_loss

    # albedo grad loss
    def get_albedo_grad_loss(self, albedo_values, albedo_gt, network_object_mask, object_mask):
        if (network_object_mask & object_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float()
        sampling_num =  int(albedo_values.size()[0] / 3)
        # split mask
        final_mask_left = network_object_mask[:sampling_num] & object_mask[:sampling_num]
        final_mask_down = network_object_mask[sampling_num:2*sampling_num] & object_mask[sampling_num:2*sampling_num]
        final_mask_origin = network_object_mask[2*sampling_num:] & object_mask[2*sampling_num:]
        valid_mask = final_mask_left & final_mask_down & final_mask_origin
        # split albedo_values
        albedo_values_left = albedo_values[:sampling_num]
        albedo_values_left = albedo_values_left[valid_mask]
        albedo_values_down = albedo_values[sampling_num:2*sampling_num]
        albedo_values_down = albedo_values_down[valid_mask]
        albedo_values_origin = albedo_values[2*sampling_num:]
        albedo_values_origin = albedo_values_origin[valid_mask]
        # split albedo_gt
        albedo_gt = albedo_gt.reshape(-1, 3)
        albedo_gt_left = albedo_gt[:sampling_num]
        albedo_gt_left = albedo_gt_left[valid_mask]
        albedo_gt_down = albedo_gt[sampling_num:2*sampling_num]
        albedo_gt_down = albedo_gt_down[valid_mask]
        albedo_gt_origin = albedo_gt[2*sampling_num:]
        albedo_gt_origin = albedo_gt_origin[valid_mask]

        albedo_values_grad_y = albedo_values_origin - albedo_values_down
        albedo_values_grad_x = albedo_values_origin - albedo_values_left

        albedo_gt_grad_y = albedo_gt_origin - albedo_gt_down
        albedo_gt_grad_x = albedo_gt_origin - albedo_gt_left

        albedo_grad_loss = torch.sum(torch.abs(albedo_values_grad_y - albedo_gt_grad_y) + torch.abs(albedo_values_grad_x - albedo_gt_grad_x)) / sampling_num
        return albedo_grad_loss

    # normal loss
    def get_normal_loss(self,normal_values, normal_gt, network_object_mask, object_mask):
        if (network_object_mask & object_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float(), torch.tensor(0.0).cuda().float()

        # shape [1,-1,3]
        normal_values = normal_values[network_object_mask & object_mask]
         # shape [1,-1,3]
        normal_gt = normal_gt.reshape(-1, 3)[network_object_mask & object_mask]
        normal_values = torch.nn.functional.normalize(normal_values, p=2, dim=-1)
        normal_gt = torch.nn.functional.normalize(normal_gt, p=2, dim=-1)
        l1 = torch.sum(torch.abs(normal_values - normal_gt)) / float(object_mask.shape[0])
        cos = torch.sum((1. - torch.sum(normal_values * normal_gt, dim = -1)))/ float(object_mask.shape[0])
        return l1, cos
    

    def get_eikonal_loss(self, grad_theta):
        if grad_theta.shape[0] == 0:
            return torch.tensor(0.0).cuda().float()

        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss

    def get_mask_loss(self, sdf_output, network_object_mask, object_mask):
        mask = ~(network_object_mask & object_mask)
        if mask.sum() == 0:
            return torch.tensor(0.0).cuda().float()
        sdf_pred = -self.alpha * sdf_output[mask]
        gt = object_mask[mask].float()
        mask_loss = (1 / self.alpha) * F.binary_cross_entropy_with_logits(
            sdf_pred.squeeze(), gt, reduction='sum') / float(object_mask.shape[0])
        return mask_loss

    def get_latent_smooth_loss(self, model_outputs):
        d_diff = model_outputs['diffuse_albedo']
        d_rough = model_outputs['roughness'][..., 0]
        d_xi_diff = model_outputs['random_xi_diffuse_albedo']
        d_xi_rough = model_outputs['random_xi_roughness'][..., 0]
        loss = nn.L1Loss()(d_diff, d_xi_diff) + nn.L1Loss()(d_rough, d_xi_rough) 
        return loss 
    
    def kl_divergence(self, rho, rho_hat):
        rho_hat = torch.mean(torch.sigmoid(rho_hat), 0)
        rho = torch.tensor([rho] * len(rho_hat)).cuda()
        return torch.mean(rho * torch.log(rho/rho_hat) + (1 - rho) * torch.log((1 - rho)/(1 - rho_hat)))

    def get_kl_loss(self, model, points):
        loss = 0
        embed_fn, _ = get_embedder(self.brdf_multires)
        values = embed_fn(points)

        for i in range(len(model.brdf_encoder_layer)):
            values = model.brdf_encoder_layer[i](values)

        loss += self.kl_divergence(0.05, values)

        return loss

    def forward(self, model_outputs, ground_truth, mat_model=None, train_idr=False):
        rgb_gt = ground_truth['rgb'].cuda()
        network_object_mask = model_outputs['network_object_mask']
        object_mask = model_outputs['object_mask']

        if train_idr:
            idr_rgb_loss = self.get_rgb_loss(model_outputs['idr_rgb'], rgb_gt, network_object_mask, object_mask)
            normal_gt = ground_truth['mono_normal'].cuda()
            normal_l1_loss, normal_cos_loss = self.get_normal_loss(model_outputs['normals'], normal_gt, network_object_mask, object_mask)
            mask_loss = self.get_mask_loss(model_outputs['sdf_output'], network_object_mask, object_mask)
            eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])
            loss = self.idr_rgb_weight * idr_rgb_loss + \
                   self.eikonal_weight * eikonal_loss + \
                   self.mask_weight * mask_loss + \
                   self.normal_l1_weight * normal_l1_loss + \
                   self.normal_cos_weight * normal_cos_loss
            
            output = {
                'eikonal_loss': eikonal_loss,
                'mask_loss': mask_loss,
                'idr_rgb_loss': idr_rgb_loss,
                'normal_l1_loss': normal_l1_loss,
                'normal_cos_loss': normal_cos_loss,
                'loss': loss}
        else:
            pred_rgb = model_outputs['sg_rgb'] + model_outputs['indir_rgb']
            sg_rgb_loss = self.get_rgb_loss(pred_rgb, rgb_gt, network_object_mask, object_mask)

            latent_smooth_loss = self.get_latent_smooth_loss(model_outputs)
            kl_loss = self.get_kl_loss(mat_model, model_outputs['points'][network_object_mask])


            if self.albedo_weight > 0:
                # compute mono albedo loss
                pred_albedo =  model_outputs['diffuse_albedo']
                albedo_gt = ground_truth['mono_albedo'].cuda()
                albedo_loss = self.get_albedo_loss(pred_albedo, albedo_gt, network_object_mask, object_mask)
                loss = self.sg_rgb_weight * sg_rgb_loss + \
                   self.kl_weight * kl_loss + \
                   self.latent_smooth_weight * latent_smooth_loss + \
                   self.albedo_weight * albedo_loss
                
                output = {
                    'sg_rgb_loss': sg_rgb_loss,
                    'kl_loss': kl_loss,
                    'latent_smooth_loss': latent_smooth_loss,
                    'albedo_loss': albedo_loss,
                    'loss': loss}
            elif self.albedo_grad_weight > 0:
                # compute mono albedo grad loss
                pred_albedo =  model_outputs['diffuse_albedo']
                albedo_gt = ground_truth['mono_albedo'].cuda()
                albedo_grad_loss = self.get_albedo_grad_loss(pred_albedo, albedo_gt, network_object_mask, object_mask)
                loss = self.sg_rgb_weight * sg_rgb_loss + \
                   self.kl_weight * kl_loss + \
                   self.latent_smooth_weight * latent_smooth_loss + \
                   self.albedo_grad_weight * albedo_grad_loss

                output = {
                    'sg_rgb_loss': sg_rgb_loss,
                    'kl_loss': kl_loss,
                    'latent_smooth_loss': latent_smooth_loss,
                    'albedo_grad_loss': albedo_grad_loss,
                    'loss': loss}
            else:
                loss = self.sg_rgb_weight * sg_rgb_loss + \
                   self.kl_weight * kl_loss + \
                   self.latent_smooth_weight * latent_smooth_loss 
                output = {
                    'sg_rgb_loss': sg_rgb_loss,
                    'kl_loss': kl_loss,
                    'latent_smooth_loss': latent_smooth_loss,
                    'loss': loss}                              

        return output


def query_indir_illum(lgtSGs, sample_dirs):
    nsamp = sample_dirs.shape[1]
    nlobe = lgtSGs.shape[1]
    lgtSGs = lgtSGs.unsqueeze(-3).expand(-1, nsamp, -1, -1)
    sample_dirs = sample_dirs.unsqueeze(-2).expand(-1, -1, nlobe, -1)
    
    lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True))
    lgtSGLambdas = lgtSGs[..., 3:4]
    lgtSGMus = lgtSGs[..., -3:]  # positive values

    pred_radiance = lgtSGMus * torch.exp(
        lgtSGLambdas * (torch.sum(sample_dirs * lgtSGLobes, dim=-1, keepdim=True) - 1.))
    pred_radiance = torch.sum(pred_radiance, dim=2)
    return pred_radiance


class IllumLoss(nn.Module):
    def __init__(self, loss_type='L1'):
        super().__init__()
        if loss_type == 'L1':
            print('Using L1 loss for comparing radiance!')
            self.rgb_loss = nn.L1Loss(reduction='mean')
        elif loss_type == 'L2':
            print('Using L2 loss for comparing radiance!')
            self.rgb_loss = nn.MSELoss(reduction='mean')
        else:
            raise Exception('Unknown loss_type!')

    def forward(self, model_outputs, trace_outputs):
        # compute indirect illumination loss
        points = model_outputs["points"]
        points_mask = model_outputs["network_object_mask"]
        lgtSGs = model_outputs["indirect_sgs"][points_mask]

        # compute radiance loss
        gt_radiance = trace_outputs['trace_radiance'][points_mask]

        sample_dirs = trace_outputs['sample_dirs']
        pred_radiance = query_indir_illum(lgtSGs, sample_dirs)

        radiance_loss = self.rgb_loss(gt_radiance, pred_radiance)

        # compute visibility loss
        gt_vis = (~trace_outputs['gt_vis'][points_mask]).long().reshape(-1)
        pred_vis = trace_outputs['pred_vis'][points_mask].reshape(-1, 2)

        visibility_loss = nn.CrossEntropyLoss()(pred_vis, gt_vis)

        return radiance_loss, visibility_loss