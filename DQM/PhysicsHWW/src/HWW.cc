#include "DQM/PhysicsHWW/interface/HWW.h"


//vertex 
std::vector<LorentzVector>  & HWW::vtxs_position(){
  if(!vtxs_position_isLoaded) edm::LogWarning("VariableNotSet") << "vtxs_position not loaded!";
  return(vtxs_position_);
}
std::vector<float>          & HWW::vtxs_ndof(){
  if(!vtxs_ndof_isLoaded) edm::LogWarning("VariableNotSet") << "vtxs_ndof not loaded!";
  return(vtxs_ndof_);
}
std::vector<float>          & HWW::vtxs_sumpt(){
  if(!vtxs_sumpt_isLoaded) edm::LogWarning("VariableNotSet") << "vtxs_sumpt not loaded!";
  return(vtxs_sumpt_);
}
std::vector<int>            & HWW::vtxs_isFake(){
  if(!vtxs_isFake_isLoaded) edm::LogWarning("VariableNotSet") << "vtxs_isFake not loaded!";
  return(vtxs_isFake_);
}
std::vector<float>          & HWW::vtxs_xError(){
  if(!vtxs_xError_isLoaded) edm::LogWarning("VariableNotSet") << "vtxs_xError not loaded!";
  return(vtxs_xError_);
}
std::vector<float>          & HWW::vtxs_yError(){
  if(!vtxs_yError_isLoaded) edm::LogWarning("VariableNotSet") << "vtxs_yError not loaded!";
  return(vtxs_yError_);
}
std::vector<float>          & HWW::vtxs_zError(){
  if(!vtxs_zError_isLoaded) edm::LogWarning("VariableNotSet") << "vtxs_zError not loaded!";
  return(vtxs_zError_);
}
std::vector<std::vector<float>>  & HWW::vtxs_covMatrix(){
  if(!vtxs_covMatrix_isLoaded) edm::LogWarning("VariableNotSet") << "vtxs_covMatrix not loaded!";
  return(vtxs_covMatrix_);
}

//tracks
std::vector<LorentzVector>  & HWW::trks_trk_p4(){
  if(!trks_trk_p4_isLoaded) edm::LogWarning("VariableNotSet") << "trks_trk_p4 not loaded!";
  return(trks_trk_p4_);
}
std::vector<LorentzVector>  & HWW::trks_vertex_p4(){
  if(!trks_vertex_p4_isLoaded) edm::LogWarning("VariableNotSet") << "trks_vertex_p4 not loaded!";
  return(trks_vertex_p4_);
}
std::vector<float>          & HWW::trks_chi2(){
  if(!trks_chi2_isLoaded) edm::LogWarning("VariableNotSet") << "trks_chi2 not loaded!";
  return(trks_chi2_);
}
std::vector<float>          & HWW::trks_ndof(){
  if(!trks_ndof_isLoaded) edm::LogWarning("VariableNotSet") << "trks_ndof not loaded!";
  return(trks_ndof_);
}
std::vector<float>          & HWW::trks_d0(){
  if(!trks_d0_isLoaded) edm::LogWarning("VariableNotSet") << "trks_d0 not loaded!";
  return(trks_d0_);
}
std::vector<int>            & HWW::trks_nlayers(){
  if(!trks_nlayers_isLoaded) edm::LogWarning("VariableNotSet") << "trks_nlayers not loaded!";
  return(trks_nlayers_);
}
std::vector<int>            & HWW::trks_valid_pixelhits(){
  if(!trks_valid_pixelhits_isLoaded) edm::LogWarning("VariableNotSet") << "trks_valid_pixelhits not loaded!";
  return(trks_valid_pixelhits_);
}
std::vector<float>          & HWW::trks_z0(){
  if(!trks_z0_isLoaded) edm::LogWarning("VariableNotSet") << "trks_z0 not loaded!";
  return(trks_z0_);
}
std::vector<float>          & HWW::trks_z0Err(){
  if(!trks_z0Err_isLoaded) edm::LogWarning("VariableNotSet") << "trks_z0Err not loaded!";
  return(trks_z0Err_);
}
std::vector<float>          & HWW::trks_etaErr(){
  if(!trks_etaErr_isLoaded) edm::LogWarning("VariableNotSet") << "trks_etaErr not loaded!";
  return(trks_etaErr_);
}
std::vector<float>          & HWW::trks_d0Err(){
  if(!trks_d0Err_isLoaded) edm::LogWarning("VariableNotSet") << "trks_d0Err not loaded!";
  return(trks_d0Err_);
}
std::vector<float>          & HWW::trks_phiErr(){
  if(!trks_phiErr_isLoaded) edm::LogWarning("VariableNotSet") << "trks_phiErr not loaded!";
  return(trks_phiErr_);
}
std::vector<float>          & HWW::trks_d0phiCov(){
  if(!trks_d0phiCov_isLoaded) edm::LogWarning("VariableNotSet") << "trks_d0phiCov not loaded!";
  return(trks_d0phiCov_);
}
std::vector<int>            & HWW::trks_qualityMask(){
  if(!trks_qualityMask_isLoaded) edm::LogWarning("VariableNotSet") << "trks_qualityMask not loaded!";
  return(trks_qualityMask_);
}
std::vector<int>            & HWW::trks_charge(){
  if(!trks_charge_isLoaded) edm::LogWarning("VariableNotSet") << "trks_charge not loaded!";
  return(trks_charge_);
}

//electrons
std::vector<LorentzVector>  & HWW::els_p4(){
  if(!els_p4_isLoaded) edm::LogWarning("VariableNotSet") << "els_p4 not loaded!";
  return(els_p4_);
}
std::vector<LorentzVector>  & HWW::els_trk_p4(){
  if(!els_trk_p4_isLoaded) edm::LogWarning("VariableNotSet") << "els_trk_p4 not loaded!";
  return(els_trk_p4_);
}
std::vector<LorentzVector>  & HWW::els_vertex_p4(){
  if(!els_vertex_p4_isLoaded) edm::LogWarning("VariableNotSet") << "els_vertex_p4 not loaded!";
  return(els_vertex_p4_);
}
std::vector<float>          & HWW::els_lh(){
  if(!els_lh_isLoaded) edm::LogWarning("VariableNotSet") << "els_lh not loaded!";
  return(els_lh_);
}
std::vector<float>          & HWW::els_etaSC(){
  if(!els_etaSC_isLoaded) edm::LogWarning("VariableNotSet") << "els_etaSC not loaded!";
  return(els_etaSC_);
}
std::vector<float>          & HWW::els_sigmaIEtaIEta(){
  if(!els_sigmaIEtaIEta_isLoaded) edm::LogWarning("VariableNotSet") << "els_sigmaIEtaIEta not loaded!";
  return(els_sigmaIEtaIEta_);
}
std::vector<float>          & HWW::els_dEtaIn(){
  if(!els_dEtaIn_isLoaded) edm::LogWarning("VariableNotSet") << "els_dEtaIn not loaded!";
  return(els_dEtaIn_);
}
std::vector<float>          & HWW::els_dPhiIn(){
  if(!els_dPhiIn_isLoaded) edm::LogWarning("VariableNotSet") << "els_dPhiIn not loaded!";
  return(els_dPhiIn_);
}
std::vector<float>          & HWW::els_hOverE(){
  if(!els_hOverE_isLoaded) edm::LogWarning("VariableNotSet") << "els_hOverE not loaded!";
  return(els_hOverE_);
}
std::vector<float>          & HWW::els_tkIso(){
  if(!els_tkIso_isLoaded) edm::LogWarning("VariableNotSet") << "els_tkIso not loaded!";
  return(els_tkIso_);
}
std::vector<float>          & HWW::els_d0corr(){
  if(!els_d0corr_isLoaded) edm::LogWarning("VariableNotSet") << "els_d0corr not loaded!";
  return(els_d0corr_);
}
std::vector<float>          & HWW::els_d0(){
  if(!els_d0_isLoaded) edm::LogWarning("VariableNotSet") << "els_d0 not loaded!";
  return(els_d0_);
}
std::vector<float>          & HWW::els_z0corr(){
  if(!els_z0corr_isLoaded) edm::LogWarning("VariableNotSet") << "els_z0corr not loaded!";
  return(els_z0corr_);
}
std::vector<float>          & HWW::els_fbrem(){
  if(!els_fbrem_isLoaded) edm::LogWarning("VariableNotSet") << "els_fbrem not loaded!";
  return(els_fbrem_);
}
std::vector<float>          & HWW::els_eOverPIn(){
  if(!els_eOverPIn_isLoaded) edm::LogWarning("VariableNotSet") << "els_eOverPIn not loaded!";
  return(els_eOverPIn_);
}
std::vector<float>          & HWW::els_eSeedOverPOut(){
  if(!els_eSeedOverPOut_isLoaded) edm::LogWarning("VariableNotSet") << "els_eSeedOverPOut not loaded!";
  return(els_eSeedOverPOut_);
}
std::vector<float>          & HWW::els_eSeedOverPIn(){
  if(!els_eSeedOverPIn_isLoaded) edm::LogWarning("VariableNotSet") << "els_eSeedOverPIn not loaded!";
  return(els_eSeedOverPIn_);
}
std::vector<float>          & HWW::els_sigmaIPhiIPhi(){
  if(!els_sigmaIPhiIPhi_isLoaded) edm::LogWarning("VariableNotSet") << "els_sigmaIPhiIPhi not loaded!";
  return(els_sigmaIPhiIPhi_);
}
std::vector<float>          & HWW::els_eSC(){
  if(!els_eSC_isLoaded) edm::LogWarning("VariableNotSet") << "els_eSC not loaded!";
  return(els_eSC_);
}
std::vector<float>          & HWW::els_ip3d(){
  if(!els_ip3d_isLoaded) edm::LogWarning("VariableNotSet") << "els_ip3d not loaded!";
  return(els_ip3d_);
}
std::vector<float>          & HWW::els_ip3derr(){
  if(!els_ip3derr_isLoaded) edm::LogWarning("VariableNotSet") << "els_ip3derr not loaded!";
  return(els_ip3derr_);
}
std::vector<float>          & HWW::els_chi2(){
  if(!els_chi2_isLoaded) edm::LogWarning("VariableNotSet") << "els_chi2 not loaded!";
  return(els_chi2_);
}
std::vector<float>          & HWW::els_ndof(){
  if(!els_ndof_isLoaded) edm::LogWarning("VariableNotSet") << "els_ndof not loaded!";
  return(els_ndof_);
}
std::vector<float>          & HWW::els_dEtaOut(){
  if(!els_dEtaOut_isLoaded) edm::LogWarning("VariableNotSet") << "els_dEtaOut not loaded!";
  return(els_dEtaOut_);
}
std::vector<float>          & HWW::els_dPhiOut(){
  if(!els_dPhiOut_isLoaded) edm::LogWarning("VariableNotSet") << "els_dPhiOut not loaded!";
  return(els_dPhiOut_);
}
std::vector<float>          & HWW::els_eSCRaw(){
  if(!els_eSCRaw_isLoaded) edm::LogWarning("VariableNotSet") << "els_eSCRaw not loaded!";
  return(els_eSCRaw_);
}
std::vector<float>          & HWW::els_etaSCwidth(){
  if(!els_etaSCwidth_isLoaded) edm::LogWarning("VariableNotSet") << "els_etaSCwidth not loaded!";
  return(els_etaSCwidth_);
}
std::vector<float>          & HWW::els_phiSCwidth(){
  if(!els_phiSCwidth_isLoaded) edm::LogWarning("VariableNotSet") << "els_phiSCwidth not loaded!";
  return(els_phiSCwidth_);
}
std::vector<float>          & HWW::els_eSCPresh(){
  if(!els_eSCPresh_isLoaded) edm::LogWarning("VariableNotSet") << "els_eSCPresh not loaded!";
  return(els_eSCPresh_);
}
std::vector<float>          & HWW::els_iso03_pf_ch(){
  if(!els_iso03_pf_ch_isLoaded) edm::LogWarning("VariableNotSet") << "els_iso03_pf_ch not loaded!";
  return(els_iso03_pf_ch_);
}
std::vector<float>          & HWW::els_iso03_pf_nhad05(){
  if(!els_iso03_pf_nhad05_isLoaded) edm::LogWarning("VariableNotSet") << "els_iso03_pf_nhad05 not loaded!";
  return(els_iso03_pf_nhad05_);
}
std::vector<float>          & HWW::els_iso03_pf_gamma05(){
  if(!els_iso03_pf_gamma05_isLoaded) edm::LogWarning("VariableNotSet") << "els_iso03_pf_gamma05 not loaded!";
  return(els_iso03_pf_gamma05_);
}
std::vector<float>          & HWW::els_iso04_pf_ch(){
  if(!els_iso04_pf_ch_isLoaded) edm::LogWarning("VariableNotSet") << "els_iso04_pf_ch not loaded!";
  return(els_iso04_pf_ch_);
}
std::vector<float>          & HWW::els_iso04_pf_nhad05(){
  if(!els_iso04_pf_nhad05_isLoaded) edm::LogWarning("VariableNotSet") << "els_iso04_pf_nhad05 not loaded!";
  return(els_iso04_pf_nhad05_);
}
std::vector<float>          & HWW::els_iso04_pf_gamma05(){
  if(!els_iso04_pf_gamma05_isLoaded) edm::LogWarning("VariableNotSet") << "els_iso04_pf_gamma05 not loaded!";
  return(els_iso04_pf_gamma05_);
}
std::vector<float>          & HWW::els_e5x5(){
  if(!els_e5x5_isLoaded) edm::LogWarning("VariableNotSet") << "els_e5x5 not loaded!";
  return(els_e5x5_);
}
std::vector<float>          & HWW::els_e1x5(){
  if(!els_e1x5_isLoaded) edm::LogWarning("VariableNotSet") << "els_e1x5 not loaded!";
  return(els_e1x5_);
}
std::vector<float>          & HWW::els_e3x3(){
  if(!els_e3x3_isLoaded) edm::LogWarning("VariableNotSet") << "els_e3x3 not loaded!";
  return(els_e3x3_);
}
std::vector<float>          & HWW::els_ecalEnergy(){
  if(!els_ecalEnergy_isLoaded) edm::LogWarning("VariableNotSet") << "els_ecalEnergy not loaded!";
  return(els_ecalEnergy_);
}
std::vector<float>          & HWW::els_eOverPOut(){
  if(!els_eOverPOut_isLoaded) edm::LogWarning("VariableNotSet") << "els_eOverPOut not loaded!";
  return(els_eOverPOut_);
}
std::vector<float>          & HWW::els_ecalIso(){
  if(!els_ecalIso_isLoaded) edm::LogWarning("VariableNotSet") << "els_ecalIso not loaded!";
  return(els_ecalIso_);
}
std::vector<float>          & HWW::els_hcalIso(){
  if(!els_hcalIso_isLoaded) edm::LogWarning("VariableNotSet") << "els_hcalIso not loaded!";
  return(els_hcalIso_);
}
std::vector<float>          & HWW::els_trkshFrac(){
  if(!els_trkshFrac_isLoaded) edm::LogWarning("VariableNotSet") << "els_trkshFrac not loaded!";
  return(els_trkshFrac_);
}
std::vector<float>          & HWW::els_conv_dist(){
  if(!els_conv_dist_isLoaded) edm::LogWarning("VariableNotSet") << "els_conv_dist not loaded!";
  return(els_conv_dist_);
}
std::vector<float>          & HWW::els_conv_dcot(){
  if(!els_conv_dcot_isLoaded) edm::LogWarning("VariableNotSet") << "els_conv_dcot not loaded!";
  return(els_conv_dcot_);
}
std::vector<float>          & HWW::els_conv_old_dist(){
  if(!els_conv_old_dist_isLoaded) edm::LogWarning("VariableNotSet") << "els_conv_old_dist not loaded!";
  return(els_conv_old_dist_);
}
std::vector<float>          & HWW::els_conv_old_dcot(){
  if(!els_conv_old_dcot_isLoaded) edm::LogWarning("VariableNotSet") << "els_conv_old_dcot not loaded!";
  return(els_conv_old_dcot_);
}
std::vector<float>          & HWW::els_iso04_pf2012_ch(){
  if(!els_iso04_pf2012_ch_isLoaded) edm::LogWarning("VariableNotSet") << "els_iso04_pf2012_ch not loaded!";
  return(els_iso04_pf2012_ch_);
}
std::vector<float>          & HWW::els_iso04_pf2012_em(){
  if(!els_iso04_pf2012_em_isLoaded) edm::LogWarning("VariableNotSet") << "els_iso04_pf2012_em not loaded!";
  return(els_iso04_pf2012_em_);
}
std::vector<float>          & HWW::els_iso04_pf2012_nh(){
  if(!els_iso04_pf2012_nh_isLoaded) edm::LogWarning("VariableNotSet") << "els_iso04_pf2012_nh not loaded!";
  return(els_iso04_pf2012_nh_);
}
std::vector<float>          & HWW::els_iso03_pf2012_ch(){
  if(!els_iso03_pf2012_ch_isLoaded) edm::LogWarning("VariableNotSet") << "els_iso03_pf2012_ch not loaded!";
  return(els_iso03_pf2012_ch_);
}
std::vector<float>          & HWW::els_iso03_pf2012_em(){
  if(!els_iso03_pf2012_em_isLoaded) edm::LogWarning("VariableNotSet") << "els_iso03_pf2012_em not loaded!";
  return(els_iso03_pf2012_em_);
}
std::vector<float>          & HWW::els_iso03_pf2012_nh(){
  if(!els_iso03_pf2012_nh_isLoaded) edm::LogWarning("VariableNotSet") << "els_iso03_pf2012_nh not loaded!";
  return(els_iso03_pf2012_nh_);
}
std::vector<float>          & HWW::els_ecalIso04(){
  if(!els_ecalIso04_isLoaded) edm::LogWarning("VariableNotSet") << "els_ecalIso04 not loaded!";
  return(els_ecalIso04_);
}
std::vector<float>          & HWW::els_hcalIso04(){
  if(!els_hcalIso04_isLoaded) edm::LogWarning("VariableNotSet") << "els_hcalIso04 not loaded!";
  return(els_hcalIso04_);
}
std::vector<int>            & HWW::els_nSeed(){
  if(!els_nSeed_isLoaded) edm::LogWarning("VariableNotSet") << "els_nSeed not loaded!";
  return(els_nSeed_);
}
std::vector<int>            & HWW::els_scindex(){
  if(!els_scindex_isLoaded) edm::LogWarning("VariableNotSet") << "els_scindex not loaded!";
  return(els_scindex_);
}
std::vector<int>            & HWW::els_charge(){
  if(!els_charge_isLoaded) edm::LogWarning("VariableNotSet") << "els_charge not loaded!";
  return(els_charge_);
}
std::vector<int>            & HWW::els_gsftrkidx(){
  if(!els_gsftrkidx_isLoaded) edm::LogWarning("VariableNotSet") << "els_gsftrkidx not loaded!";
  return(els_gsftrkidx_);
}
std::vector<int>            & HWW::els_exp_innerlayers(){
  if(!els_exp_innerlayers_isLoaded) edm::LogWarning("VariableNotSet") << "els_exp_innerlayers not loaded!";
  return(els_exp_innerlayers_);
}
std::vector<int>            & HWW::els_trkidx(){
  if(!els_trkidx_isLoaded) edm::LogWarning("VariableNotSet") << "els_trkidx not loaded!";
  return(els_trkidx_);
}
std::vector<int>            & HWW::els_type(){
  if(!els_type_isLoaded) edm::LogWarning("VariableNotSet") << "els_type not loaded!";
  return(els_type_);
}
std::vector<int>            & HWW::els_fiduciality(){
  if(!els_fiduciality_isLoaded) edm::LogWarning("VariableNotSet") << "els_fiduciality not loaded!";
  return(els_fiduciality_);
}
std::vector<int>            & HWW::els_sccharge(){
  if(!els_sccharge_isLoaded) edm::LogWarning("VariableNotSet") << "els_sccharge not loaded!";
  return(els_sccharge_);
}
std::vector<int>            & HWW::els_trk_charge(){
  if(!els_trk_charge_isLoaded) edm::LogWarning("VariableNotSet") << "els_trk_charge not loaded!";
  return(els_trk_charge_);
}
std::vector<int>            & HWW::els_closestMuon(){
  if(!els_closestMuon_isLoaded) edm::LogWarning("VariableNotSet") << "els_closestMuon not loaded!";
  return(els_closestMuon_);
}

//muons
std::vector<LorentzVector>  & HWW::mus_p4(){
  if(!mus_p4_isLoaded) edm::LogWarning("VariableNotSet") << "mus_p4 not loaded!";
  return(mus_p4_);
}
std::vector<LorentzVector>  & HWW::mus_trk_p4(){
  if(!mus_trk_p4_isLoaded) edm::LogWarning("VariableNotSet") << "mus_trk_p4 not loaded!";
  return(mus_trk_p4_);
}
std::vector<LorentzVector>  & HWW::mus_vertex_p4(){
  if(!mus_vertex_p4_isLoaded) edm::LogWarning("VariableNotSet") << "mus_vertex_p4 not loaded!";
  return(mus_vertex_p4_);
}
std::vector<LorentzVector>  & HWW::mus_sta_p4(){
  if(!mus_sta_p4_isLoaded) edm::LogWarning("VariableNotSet") << "mus_sta_p4 not loaded!";
  return(mus_sta_p4_);
}
std::vector<float>          & HWW::mus_gfit_chi2(){
  if(!mus_gfit_chi2_isLoaded) edm::LogWarning("VariableNotSet") << "mus_gfit_chi2 not loaded!";
  return(mus_gfit_chi2_);
}
std::vector<float>          & HWW::mus_gfit_ndof(){
  if(!mus_gfit_ndof_isLoaded) edm::LogWarning("VariableNotSet") << "mus_gfit_ndof not loaded!";
  return(mus_gfit_ndof_);
}
std::vector<float>          & HWW::mus_ptErr(){
  if(!mus_ptErr_isLoaded) edm::LogWarning("VariableNotSet") << "mus_ptErr not loaded!";
  return(mus_ptErr_);
}
std::vector<float>          & HWW::mus_trkKink(){
  if(!mus_trkKink_isLoaded) edm::LogWarning("VariableNotSet") << "mus_trkKink not loaded!";
  return(mus_trkKink_);
}
std::vector<float>          & HWW::mus_d0corr(){
  if(!mus_d0corr_isLoaded) edm::LogWarning("VariableNotSet") << "mus_d0corr not loaded!";
  return(mus_d0corr_);
}
std::vector<float>          & HWW::mus_d0(){
  if(!mus_d0_isLoaded) edm::LogWarning("VariableNotSet") << "mus_d0 not loaded!";
  return(mus_d0_);
}
std::vector<float>          & HWW::mus_z0corr(){
  if(!mus_z0corr_isLoaded) edm::LogWarning("VariableNotSet") << "mus_z0corr not loaded!";
  return(mus_z0corr_);
}
std::vector<float>          & HWW::mus_chi2(){
  if(!mus_chi2_isLoaded) edm::LogWarning("VariableNotSet") << "mus_chi2 not loaded!";
  return(mus_chi2_);
}
std::vector<float>          & HWW::mus_ndof(){
  if(!mus_ndof_isLoaded) edm::LogWarning("VariableNotSet") << "mus_ndof not loaded!";
  return(mus_ndof_);
}
std::vector<float>          & HWW::mus_ip3d(){
  if(!mus_ip3d_isLoaded) edm::LogWarning("VariableNotSet") << "mus_ip3d not loaded!";
  return(mus_ip3d_);
}
std::vector<float>          & HWW::mus_ip3derr(){
  if(!mus_ip3derr_isLoaded) edm::LogWarning("VariableNotSet") << "mus_ip3derr not loaded!";
  return(mus_ip3derr_);
}
std::vector<float>          & HWW::mus_segmCompatibility(){
  if(!mus_segmCompatibility_isLoaded) edm::LogWarning("VariableNotSet") << "mus_segmCompatibility not loaded!";
  return(mus_segmCompatibility_);
}
std::vector<float>          & HWW::mus_caloCompatibility(){
  if(!mus_caloCompatibility_isLoaded) edm::LogWarning("VariableNotSet") << "mus_caloCompatibility not loaded!";
  return(mus_caloCompatibility_);
}
std::vector<float>          & HWW::mus_e_had(){
  if(!mus_e_had_isLoaded) edm::LogWarning("VariableNotSet") << "mus_e_had not loaded!";
  return(mus_e_had_);
}
std::vector<float>          & HWW::mus_e_ho(){
  if(!mus_e_ho_isLoaded) edm::LogWarning("VariableNotSet") << "mus_e_ho not loaded!";
  return(mus_e_ho_);
}
std::vector<float>          & HWW::mus_e_em(){
  if(!mus_e_em_isLoaded) edm::LogWarning("VariableNotSet") << "mus_e_em not loaded!";
  return(mus_e_em_);
}
std::vector<float>          & HWW::mus_e_hadS9(){
  if(!mus_e_hadS9_isLoaded) edm::LogWarning("VariableNotSet") << "mus_e_hadS9 not loaded!";
  return(mus_e_hadS9_);
}
std::vector<float>          & HWW::mus_e_hoS9(){
  if(!mus_e_hoS9_isLoaded) edm::LogWarning("VariableNotSet") << "mus_e_hoS9 not loaded!";
  return(mus_e_hoS9_);
}
std::vector<float>          & HWW::mus_e_emS9(){
  if(!mus_e_emS9_isLoaded) edm::LogWarning("VariableNotSet") << "mus_e_emS9 not loaded!";
  return(mus_e_emS9_);
}
std::vector<float>          & HWW::mus_iso03_sumPt(){
  if(!mus_iso03_sumPt_isLoaded) edm::LogWarning("VariableNotSet") << "mus_iso03_sumPt not loaded!";
  return(mus_iso03_sumPt_);
}
std::vector<float>          & HWW::mus_iso03_emEt(){
  if(!mus_iso03_emEt_isLoaded) edm::LogWarning("VariableNotSet") << "mus_iso03_emEt not loaded!";
  return(mus_iso03_emEt_);
}
std::vector<float>          & HWW::mus_iso03_hadEt(){
  if(!mus_iso03_hadEt_isLoaded) edm::LogWarning("VariableNotSet") << "mus_iso03_hadEt not loaded!";
  return(mus_iso03_hadEt_);
}
std::vector<float>          & HWW::mus_iso05_sumPt(){
  if(!mus_iso05_sumPt_isLoaded) edm::LogWarning("VariableNotSet") << "mus_iso05_sumPt not loaded!";
  return(mus_iso05_sumPt_);
}
std::vector<float>          & HWW::mus_iso05_emEt(){
  if(!mus_iso05_emEt_isLoaded) edm::LogWarning("VariableNotSet") << "mus_iso05_emEt not loaded!";
  return(mus_iso05_emEt_);
}
std::vector<float>          & HWW::mus_iso05_hadEt(){
  if(!mus_iso05_hadEt_isLoaded) edm::LogWarning("VariableNotSet") << "mus_iso05_hadEt not loaded!";
  return(mus_iso05_hadEt_);
}
std::vector<float>          & HWW::mus_sta_d0(){
  if(!mus_sta_d0_isLoaded) edm::LogWarning("VariableNotSet") << "mus_sta_d0 not loaded!";
  return(mus_sta_d0_);
}
std::vector<float>          & HWW::mus_sta_z0corr(){
  if(!mus_sta_z0corr_isLoaded) edm::LogWarning("VariableNotSet") << "mus_sta_z0corr not loaded!";
  return(mus_sta_z0corr_);
}
std::vector<float>          & HWW::mus_isoR03_pf_ChargedHadronPt(){
  if(!mus_isoR03_pf_ChargedHadronPt_isLoaded) edm::LogWarning("VariableNotSet") << "mus_isoR03_pf_ChargedHadronPt not loaded!";
  return(mus_isoR03_pf_ChargedHadronPt_);
}
std::vector<float>          & HWW::mus_isoR03_pf_NeutralHadronEt(){
  if(!mus_isoR03_pf_NeutralHadronEt_isLoaded) edm::LogWarning("VariableNotSet") << "mus_isoR03_pf_NeutralHadronEt not loaded!";
  return(mus_isoR03_pf_NeutralHadronEt_);
}
std::vector<float>          & HWW::mus_isoR03_pf_PhotonEt(){
  if(!mus_isoR03_pf_PhotonEt_isLoaded) edm::LogWarning("VariableNotSet") << "mus_isoR03_pf_PhotonEt not loaded!";
  return(mus_isoR03_pf_PhotonEt_);
}
std::vector<float>          & HWW::mus_isoR03_pf_PUPt(){
  if(!mus_isoR03_pf_PUPt_isLoaded) edm::LogWarning("VariableNotSet") << "mus_isoR03_pf_PUPt not loaded!";
  return(mus_isoR03_pf_PUPt_);
}
std::vector<float>          & HWW::mus_iso_ecalvetoDep(){
  if(!mus_iso_ecalvetoDep_isLoaded) edm::LogWarning("VariableNotSet") << "mus_iso_ecalvetoDep not loaded!";
  return(mus_iso_ecalvetoDep_);
}
std::vector<float>          & HWW::mus_iso_hcalvetoDep(){
  if(!mus_iso_hcalvetoDep_isLoaded) edm::LogWarning("VariableNotSet") << "mus_iso_hcalvetoDep not loaded!";
  return(mus_iso_hcalvetoDep_);
}
std::vector<int>            & HWW::mus_gfit_validSTAHits(){
  if(!mus_gfit_validSTAHits_isLoaded) edm::LogWarning("VariableNotSet") << "mus_gfit_validSTAHits not loaded!";
  return(mus_gfit_validSTAHits_);
}
std::vector<int>            & HWW::mus_numberOfMatchedStations(){
  if(!mus_numberOfMatchedStations_isLoaded) edm::LogWarning("VariableNotSet") << "mus_numberOfMatchedStations not loaded!";
  return(mus_numberOfMatchedStations_);
}
std::vector<int>            & HWW::mus_pfmusidx(){
  if(!mus_pfmusidx_isLoaded) edm::LogWarning("VariableNotSet") << "mus_pfmusidx not loaded!";
  return(mus_pfmusidx_);
}
std::vector<int>            & HWW::mus_charge(){
  if(!mus_charge_isLoaded) edm::LogWarning("VariableNotSet") << "mus_charge not loaded!";
  return(mus_charge_);
}
std::vector<int>            & HWW::mus_validHits(){
  if(!mus_validHits_isLoaded) edm::LogWarning("VariableNotSet") << "mus_validHits not loaded!";
  return(mus_validHits_);
}
std::vector<int>            & HWW::mus_trkidx(){
  if(!mus_trkidx_isLoaded) edm::LogWarning("VariableNotSet") << "mus_trkidx not loaded!";
  return(mus_trkidx_);
}
std::vector<int>            & HWW::mus_pid_PFMuon(){
  if(!mus_pid_PFMuon_isLoaded) edm::LogWarning("VariableNotSet") << "mus_pid_PFMuon not loaded!";
  return(mus_pid_PFMuon_);
}
std::vector<int>            & HWW::mus_pid_TMLastStationTight(){
  if(!mus_pid_TMLastStationTight_isLoaded) edm::LogWarning("VariableNotSet") << "mus_pid_TMLastStationTight not loaded!";
  return(mus_pid_TMLastStationTight_);
}
std::vector<int>            & HWW::mus_nmatches(){
  if(!mus_nmatches_isLoaded) edm::LogWarning("VariableNotSet") << "mus_nmatches not loaded!";
  return(mus_nmatches_);
}
std::vector<int>            & HWW::mus_goodmask(){
  if(!mus_goodmask_isLoaded) edm::LogWarning("VariableNotSet") << "mus_goodmask not loaded!";
  return(mus_goodmask_);
}
std::vector<int>            & HWW::mus_type(){
  if(!mus_type_isLoaded) edm::LogWarning("VariableNotSet") << "mus_type not loaded!";
  return(mus_type_);
}

//dilepton hypothesis
std::vector<std::vector<LorentzVector> > & HWW::hyp_jets_p4(){
  if(!hyp_jets_p4_isLoaded) edm::LogWarning("VariableNotSet") << "hyp_jets_p4 not loaded!";
  return(hyp_jets_p4_);
}
std::vector<LorentzVector>  & HWW::hyp_p4(){
  if(!hyp_p4_isLoaded) edm::LogWarning("VariableNotSet") << "hyp_p4 not loaded!";
  return(hyp_p4_);
}
std::vector<LorentzVector>  & HWW::hyp_ll_p4(){
  if(!hyp_ll_p4_isLoaded) edm::LogWarning("VariableNotSet") << "hyp_ll_p4 not loaded!";
  return(hyp_ll_p4_);
}
std::vector<LorentzVector>  & HWW::hyp_lt_p4(){
  if(!hyp_lt_p4_isLoaded) edm::LogWarning("VariableNotSet") << "hyp_lt_p4 not loaded!";
  return(hyp_lt_p4_);
}
std::vector<int>            & HWW::hyp_ll_index(){
  if(!hyp_ll_index_isLoaded) edm::LogWarning("VariableNotSet") << "hyp_ll_index not loaded!";
  return(hyp_ll_index_);
}
std::vector<int>            & HWW::hyp_lt_index(){
  if(!hyp_lt_index_isLoaded) edm::LogWarning("VariableNotSet") << "hyp_lt_index not loaded!";
  return(hyp_lt_index_);
}
std::vector<int>            & HWW::hyp_ll_id(){
  if(!hyp_ll_id_isLoaded) edm::LogWarning("VariableNotSet") << "hyp_ll_id not loaded!";
  return(hyp_ll_id_);
}
std::vector<int>            & HWW::hyp_lt_id(){
  if(!hyp_lt_id_isLoaded) edm::LogWarning("VariableNotSet") << "hyp_lt_id not loaded!";
  return(hyp_lt_id_);
}
std::vector<int>            & HWW::hyp_ll_charge(){
  if(!hyp_ll_charge_isLoaded) edm::LogWarning("VariableNotSet") << "hyp_ll_charge not loaded!";
  return(hyp_ll_charge_);
}
std::vector<int>            & HWW::hyp_lt_charge(){
  if(!hyp_lt_charge_isLoaded) edm::LogWarning("VariableNotSet") << "hyp_lt_charge not loaded!";
  return(hyp_lt_charge_);
}
std::vector<int>            & HWW::hyp_type(){
  if(!hyp_type_isLoaded) edm::LogWarning("VariableNotSet") << "hyp_type not loaded!";
  return(hyp_type_);
}

//event variables
unsigned int                & HWW::evt_run(){
  if(!evt_run_isLoaded) edm::LogWarning("VariableNotSet") << "evt_run not loaded!";
  return(evt_run_);
}
unsigned int                & HWW::evt_lumiBlock(){
  if(!evt_lumiBlock_isLoaded) edm::LogWarning("VariableNotSet") << "evt_lumiBlock not loaded!";
  return(evt_lumiBlock_);
}
unsigned int                & HWW::evt_event(){
  if(!evt_event_isLoaded) edm::LogWarning("VariableNotSet") << "evt_event not loaded!";
  return(evt_event_);
}
int                         & HWW::evt_isRealData(){
  if(!evt_isRealData_isLoaded) edm::LogWarning("VariableNotSet") << "evt_isRealData not loaded!";
  return(evt_isRealData_);
}
float                       & HWW::evt_ww_rho_vor(){
  if(!evt_ww_rho_vor_isLoaded) edm::LogWarning("VariableNotSet") << "evt_ww_rho_vor not loaded!";
  return(evt_ww_rho_vor_);
}
float                       & HWW::evt_ww_rho(){
  if(!evt_ww_rho_isLoaded) edm::LogWarning("VariableNotSet") << "evt_ww_rho not loaded!";
  return(evt_ww_rho_);
}
float                       & HWW::evt_rho(){
  if(!evt_rho_isLoaded) edm::LogWarning("VariableNotSet") << "evt_rho not loaded!";
  return(evt_rho_);
}
float                       & HWW::evt_kt6pf_foregiso_rho(){
  if(!evt_kt6pf_foregiso_rho_isLoaded) edm::LogWarning("VariableNotSet") << "evt_kt6pf_foregiso_rho not loaded!";
  return(evt_kt6pf_foregiso_rho_);
}
float                       & HWW::evt_pfmet(){
  if(!evt_pfmet_isLoaded) edm::LogWarning("VariableNotSet") << "evt_pfmet not loaded!";
  return(evt_pfmet_);
}
float                       & HWW::evt_pfmetPhi(){
  if(!evt_pfmetPhi_isLoaded) edm::LogWarning("VariableNotSet") << "evt_pfmetPhi not loaded!";
  return(evt_pfmetPhi_);
}


std::vector<float>          & HWW::convs_ndof(){
  if(!convs_ndof_isLoaded) edm::LogWarning("VariableNotSet") << "convs_ndof not loaded!";
  return(convs_ndof_);
}
std::vector<float>          & HWW::convs_chi2(){
  if(!convs_chi2_isLoaded) edm::LogWarning("VariableNotSet") << "convs_chi2 not loaded!";
  return(convs_chi2_);
}
std::vector<float>          & HWW::convs_dl(){
  if(!convs_dl_isLoaded) edm::LogWarning("VariableNotSet") << "convs_dl not loaded!";
  return(convs_dl_);
}
std::vector<int>            & HWW::convs_isConverted(){
  if(!convs_isConverted_isLoaded) edm::LogWarning("VariableNotSet") << "convs_isConverted not loaded!";
  return(convs_isConverted_);
}
std::vector<std::vector<int> >    & HWW::convs_tkalgo(){
  if(!convs_tkalgo_isLoaded) edm::LogWarning("VariableNotSet") << "convs_tkalgo not loaded!";
  return(convs_tkalgo_);
}
std::vector<std::vector<int> >    & HWW::convs_tkidx(){
  if(!convs_tkidx_isLoaded) edm::LogWarning("VariableNotSet") << "convs_tkidx not loaded!";
  return(convs_tkidx_);
}
std::vector<std::vector<int> >    & HWW::convs_nHitsBeforeVtx(){
  if(!convs_nHitsBeforeVtx_isLoaded) edm::LogWarning("VariableNotSet") << "convs_nHitsBeforeVtx not loaded!";
  return(convs_nHitsBeforeVtx_);
}
std::vector<int>            & HWW::convs_quality(){
  if(!convs_quality_isLoaded) edm::LogWarning("VariableNotSet") << "convs_quality not loaded!";
  return(convs_quality_);
}
std::vector<float>          & HWW::scs_sigmaIEtaIPhi(){
  if(!scs_sigmaIEtaIPhi_isLoaded) edm::LogWarning("VariableNotSet") << "scs_sigmaIEtaIPhi not loaded!";
  return(scs_sigmaIEtaIPhi_);
}
std::vector<LorentzVector>  & HWW::scs_pos_p4(){
  if(!scs_pos_p4_isLoaded) edm::LogWarning("VariableNotSet") << "scs_pos_p4 not loaded!";
  return(scs_pos_p4_);
}
std::vector<LorentzVector>  & HWW::gsftrks_p4(){
  if(!gsftrks_p4_isLoaded) edm::LogWarning("VariableNotSet") << "gsftrks_p4 not loaded!";
  return(gsftrks_p4_);
}
std::vector<LorentzVector>  & HWW::gsftrks_vertex_p4(){
  if(!gsftrks_vertex_p4_isLoaded) edm::LogWarning("VariableNotSet") << "gsftrks_vertex_p4 not loaded!";
  return(gsftrks_vertex_p4_);
}
std::vector<float>          & HWW::gsftrks_d0(){
  if(!gsftrks_d0_isLoaded) edm::LogWarning("VariableNotSet") << "gsftrks_d0 not loaded!";
  return(gsftrks_d0_);
}
std::vector<float>          & HWW::gsftrks_d0Err(){
  if(!gsftrks_d0Err_isLoaded) edm::LogWarning("VariableNotSet") << "gsftrks_d0Err not loaded!";
  return(gsftrks_d0Err_);
}
std::vector<float>          & HWW::gsftrks_phiErr(){
  if(!gsftrks_phiErr_isLoaded) edm::LogWarning("VariableNotSet") << "gsftrks_phiErr not loaded!";
  return(gsftrks_phiErr_);
}
std::vector<float>          & HWW::gsftrks_d0phiCov(){
  if(!gsftrks_d0phiCov_isLoaded) edm::LogWarning("VariableNotSet") << "gsftrks_d0phiCov not loaded!";
  return(gsftrks_d0phiCov_);
}
std::vector<float>          & HWW::gsftrks_z0Err(){
  if(!gsftrks_z0Err_isLoaded) edm::LogWarning("VariableNotSet") << "gsftrks_z0Err not loaded!";
  return(gsftrks_z0Err_);
}
std::vector<float>          & HWW::gsftrks_z0(){
  if(!gsftrks_z0_isLoaded) edm::LogWarning("VariableNotSet") << "gsftrks_z0 not loaded!";
  return(gsftrks_z0_);
}
std::vector<float>          & HWW::gsftrks_etaErr(){
  if(!gsftrks_etaErr_isLoaded) edm::LogWarning("VariableNotSet") << "gsftrks_etaErr not loaded!";
  return(gsftrks_etaErr_);
}
std::vector<LorentzVector>  & HWW::pfcands_p4(){
  if(!pfcands_p4_isLoaded) edm::LogWarning("VariableNotSet") << "pfcands_p4 not loaded!";
  return(pfcands_p4_);
}
std::vector<int>            & HWW::pfcands_trkidx(){
  if(!pfcands_trkidx_isLoaded) edm::LogWarning("VariableNotSet") << "pfcands_trkidx not loaded!";
  return(pfcands_trkidx_);
}
std::vector<int>            & HWW::pfcands_particleId(){
  if(!pfcands_particleId_isLoaded) edm::LogWarning("VariableNotSet") << "pfcands_particleId not loaded!";
  return(pfcands_particleId_);
}
std::vector<int>            & HWW::pfcands_pfelsidx(){
  if(!pfcands_pfelsidx_isLoaded) edm::LogWarning("VariableNotSet") << "pfcands_pfelsidx not loaded!";
  return(pfcands_pfelsidx_);
}
std::vector<int>            & HWW::pfcands_vtxidx(){
  if(!pfcands_vtxidx_isLoaded) edm::LogWarning("VariableNotSet") << "pfcands_vtxidx not loaded!";
  return(pfcands_vtxidx_);
}
std::vector<int>            & HWW::pfcands_charge(){
  if(!pfcands_charge_isLoaded) edm::LogWarning("VariableNotSet") << "pfcands_charge not loaded!";
  return(pfcands_charge_);
}
std::vector<int>            & HWW::pfels_elsidx(){
  if(!pfels_elsidx_isLoaded) edm::LogWarning("VariableNotSet") << "pfels_elsidx not loaded!";
  return(pfels_elsidx_);
}
std::vector<LorentzVector>  & HWW::pfels_p4(){
  if(!pfels_p4_isLoaded) edm::LogWarning("VariableNotSet") << "pfels_p4 not loaded!";
  return(pfels_p4_);
}
std::vector<LorentzVector>  & HWW::pfmus_p4(){
  if(!pfmus_p4_isLoaded) edm::LogWarning("VariableNotSet") << "pfmus_p4 not loaded!";
  return(pfmus_p4_);
}
std::vector<float>               & HWW::trk_met(){
  if(!trk_met_isLoaded) edm::LogWarning("VariableNotSet") << "trk_met not loaded!";
  return(trk_met_);
}
std::vector<float>               & HWW::trk_metPhi(){
  if(!trk_metPhi_isLoaded) edm::LogWarning("VariableNotSet") << "trk_metPhi not loaded!";
  return(trk_metPhi_);
}
std::vector<LorentzVector>       & HWW::pfjets_p4(){
  if(!pfjets_p4_isLoaded) edm::LogWarning("VariableNotSet") << "pfjets_p4 not loaded!";
  return(pfjets_p4_);
}
std::vector<LorentzVector>       & HWW::pfjets_corr_p4(){
  if(!pfjets_corr_p4_isLoaded) edm::LogWarning("VariableNotSet") << "pfjets_corr_p4 not loaded!";
  return(pfjets_corr_p4_);
}
std::vector<float>               & HWW::pfjets_area(){
  if(!pfjets_area_isLoaded) edm::LogWarning("VariableNotSet") << "pfjets_area not loaded!";
  return(pfjets_area_);
}
std::vector<float>               & HWW::pfjets_JEC(){
  if(!pfjets_JEC_isLoaded) edm::LogWarning("VariableNotSet") << "pfjets_JEC not loaded!";
  return(pfjets_JEC_);
}
std::vector<float>               & HWW::pfjets_mvavalue(){
  if(!pfjets_mvavalue_isLoaded) edm::LogWarning("VariableNotSet") << "pfjets_mvavalue not loaded!";
  return(pfjets_mvavalue_);
}
std::vector<float>               & HWW::pfjets_trackCountingHighEffBJetTag(){
  if(!pfjets_trackCountingHighEffBJetTag_isLoaded) edm::LogWarning("VariableNotSet") << "pfjets_trackCountingHighEffBJetTag not loaded!";
  return(pfjets_trackCountingHighEffBJetTag_);
}







void HWW::Load_vtxs_position(){
  vtxs_position_isLoaded = true;
}
void HWW::Load_vtxs_ndof(){
  vtxs_ndof_isLoaded = true;
}
void HWW::Load_vtxs_sumpt(){
  vtxs_sumpt_isLoaded = true;
}
void HWW::Load_vtxs_isFake(){
  vtxs_isFake_isLoaded = true;
}
void HWW::Load_vtxs_xError(){
  vtxs_xError_isLoaded = true;
}
void HWW::Load_vtxs_yError(){
  vtxs_yError_isLoaded = true;
}
void HWW::Load_vtxs_zError(){
  vtxs_zError_isLoaded = true;
}
void HWW::Load_vtxs_covMatrix(){
  vtxs_covMatrix_isLoaded = true;
}
void HWW::Load_trks_trk_p4(){
  trks_trk_p4_isLoaded = true;
}
void HWW::Load_trks_vertex_p4(){
  trks_vertex_p4_isLoaded = true;
}
void HWW::Load_trks_chi2(){
  trks_chi2_isLoaded = true;
}
void HWW::Load_trks_ndof(){
  trks_ndof_isLoaded = true;
}
void HWW::Load_trks_d0(){
  trks_d0_isLoaded = true;
}
void HWW::Load_trks_nlayers(){
  trks_nlayers_isLoaded = true;
}
void HWW::Load_trks_valid_pixelhits(){
  trks_valid_pixelhits_isLoaded = true;
}
void HWW::Load_trks_z0(){
  trks_z0_isLoaded = true;
}
void HWW::Load_trks_z0Err(){
  trks_z0Err_isLoaded = true;
}
void HWW::Load_trks_etaErr(){
  trks_etaErr_isLoaded = true;
}
void HWW::Load_trks_d0Err(){
  trks_d0Err_isLoaded = true;
}
void HWW::Load_trks_phiErr(){
  trks_phiErr_isLoaded = true;
}
void HWW::Load_trks_d0phiCov(){
  trks_d0phiCov_isLoaded = true;
}
void HWW::Load_trks_qualityMask(){
  trks_qualityMask_isLoaded = true;
}
void HWW::Load_trks_charge(){
  trks_charge_isLoaded = true;
}
void HWW::Load_els_p4(){
  els_p4_isLoaded = true;
}
void HWW::Load_els_trk_p4(){
  els_trk_p4_isLoaded = true;
}
void HWW::Load_els_vertex_p4(){
  els_vertex_p4_isLoaded = true;
}
void HWW::Load_els_lh(){
  els_lh_isLoaded = true;
}
void HWW::Load_els_etaSC(){
  els_etaSC_isLoaded = true;
}
void HWW::Load_els_sigmaIEtaIEta(){
  els_sigmaIEtaIEta_isLoaded = true;
}
void HWW::Load_els_dEtaIn(){
  els_dEtaIn_isLoaded = true;
}
void HWW::Load_els_dPhiIn(){
  els_dPhiIn_isLoaded = true;
}
void HWW::Load_els_hOverE(){
  els_hOverE_isLoaded = true;
}
void HWW::Load_els_tkIso(){
  els_tkIso_isLoaded = true;
}
void HWW::Load_els_d0corr(){
  els_d0corr_isLoaded = true;
}
void HWW::Load_els_d0(){
  els_d0_isLoaded = true;
}
void HWW::Load_els_z0corr(){
  els_z0corr_isLoaded = true;
}
void HWW::Load_els_fbrem(){
  els_fbrem_isLoaded = true;
}
void HWW::Load_els_eOverPIn(){
  els_eOverPIn_isLoaded = true;
}
void HWW::Load_els_eSeedOverPOut(){
  els_eSeedOverPOut_isLoaded = true;
}
void HWW::Load_els_eSeedOverPIn(){
  els_eSeedOverPIn_isLoaded = true;
}
void HWW::Load_els_sigmaIPhiIPhi(){
  els_sigmaIPhiIPhi_isLoaded = true;
}
void HWW::Load_els_eSC(){
  els_eSC_isLoaded = true;
}
void HWW::Load_els_ip3d(){
  els_ip3d_isLoaded = true;
}
void HWW::Load_els_ip3derr(){
  els_ip3derr_isLoaded = true;
}
void HWW::Load_els_chi2(){
  els_chi2_isLoaded = true;
}
void HWW::Load_els_ndof(){
  els_ndof_isLoaded = true;
}
void HWW::Load_els_dEtaOut(){
  els_dEtaOut_isLoaded = true;
}
void HWW::Load_els_dPhiOut(){
  els_dPhiOut_isLoaded = true;
}
void HWW::Load_els_eSCRaw(){
  els_eSCRaw_isLoaded = true;
}
void HWW::Load_els_etaSCwidth(){
  els_etaSCwidth_isLoaded = true;
}
void HWW::Load_els_phiSCwidth(){
  els_phiSCwidth_isLoaded = true;
}
void HWW::Load_els_eSCPresh(){
  els_eSCPresh_isLoaded = true;
}
void HWW::Load_els_iso03_pf_ch(){
  els_iso03_pf_ch_isLoaded = true;
}
void HWW::Load_els_iso03_pf_nhad05(){
  els_iso03_pf_nhad05_isLoaded = true;
}
void HWW::Load_els_iso03_pf_gamma05(){
  els_iso03_pf_gamma05_isLoaded = true;
}
void HWW::Load_els_iso04_pf_ch(){
  els_iso04_pf_ch_isLoaded = true;
}
void HWW::Load_els_iso04_pf_nhad05(){
  els_iso04_pf_nhad05_isLoaded = true;
}
void HWW::Load_els_iso04_pf_gamma05(){
  els_iso04_pf_gamma05_isLoaded = true;
}
void HWW::Load_els_e5x5(){
  els_e5x5_isLoaded = true;
}
void HWW::Load_els_e1x5(){
  els_e1x5_isLoaded = true;
}
void HWW::Load_els_e3x3(){
  els_e3x3_isLoaded = true;
}
void HWW::Load_els_ecalEnergy(){
  els_ecalEnergy_isLoaded = true;
}
void HWW::Load_els_eOverPOut(){
  els_eOverPOut_isLoaded = true;
}
void HWW::Load_els_ecalIso(){
  els_ecalIso_isLoaded = true;
}
void HWW::Load_els_hcalIso(){
  els_hcalIso_isLoaded = true;
}
void HWW::Load_els_trkshFrac(){
  els_trkshFrac_isLoaded = true;
}
void HWW::Load_els_conv_dist(){
  els_conv_dist_isLoaded = true;
}
void HWW::Load_els_conv_dcot(){
  els_conv_dcot_isLoaded = true;
}
void HWW::Load_els_conv_old_dist(){
  els_conv_old_dist_isLoaded = true;
}
void HWW::Load_els_conv_old_dcot(){
  els_conv_old_dcot_isLoaded = true;
}
void HWW::Load_els_iso04_pf2012_ch(){
  els_iso04_pf2012_ch_isLoaded = true;
}
void HWW::Load_els_iso04_pf2012_em(){
  els_iso04_pf2012_em_isLoaded = true;
}
void HWW::Load_els_iso04_pf2012_nh(){
  els_iso04_pf2012_nh_isLoaded = true;
}
void HWW::Load_els_iso03_pf2012_ch(){
  els_iso03_pf2012_ch_isLoaded = true;
}
void HWW::Load_els_iso03_pf2012_em(){
  els_iso03_pf2012_em_isLoaded = true;
}
void HWW::Load_els_iso03_pf2012_nh(){
  els_iso03_pf2012_nh_isLoaded = true;
}
void HWW::Load_els_ecalIso04(){
  els_ecalIso04_isLoaded = true;
}
void HWW::Load_els_hcalIso04(){
  els_hcalIso04_isLoaded = true;
}
void HWW::Load_els_nSeed(){
  els_nSeed_isLoaded = true;
}
void HWW::Load_els_scindex(){
  els_scindex_isLoaded = true;
}
void HWW::Load_els_charge(){
  els_charge_isLoaded = true;
}
void HWW::Load_els_gsftrkidx(){
  els_gsftrkidx_isLoaded = true;
}
void HWW::Load_els_exp_innerlayers(){
  els_exp_innerlayers_isLoaded = true;
}
void HWW::Load_els_trkidx(){
  els_trkidx_isLoaded = true;
}
void HWW::Load_els_type(){
  els_type_isLoaded = true;
}
void HWW::Load_els_fiduciality(){
  els_fiduciality_isLoaded = true;
}
void HWW::Load_els_sccharge(){
  els_sccharge_isLoaded = true;
}
void HWW::Load_els_trk_charge(){
  els_trk_charge_isLoaded = true;
}
void HWW::Load_els_closestMuon(){
  els_closestMuon_isLoaded = true;
}
void HWW::Load_mus_p4(){
  mus_p4_isLoaded = true;
}
void HWW::Load_mus_trk_p4(){
  mus_trk_p4_isLoaded = true;
}
void HWW::Load_mus_vertex_p4(){
  mus_vertex_p4_isLoaded = true;
}
void HWW::Load_mus_sta_p4(){
  mus_sta_p4_isLoaded = true;
}
void HWW::Load_mus_gfit_chi2(){
  mus_gfit_chi2_isLoaded = true;
}
void HWW::Load_mus_gfit_ndof(){
  mus_gfit_ndof_isLoaded = true;
}
void HWW::Load_mus_ptErr(){
  mus_ptErr_isLoaded = true;
}
void HWW::Load_mus_trkKink(){
  mus_trkKink_isLoaded = true;
}
void HWW::Load_mus_d0corr(){
  mus_d0corr_isLoaded = true;
}
void HWW::Load_mus_d0(){
  mus_d0_isLoaded = true;
}
void HWW::Load_mus_z0corr(){
  mus_z0corr_isLoaded = true;
}
void HWW::Load_mus_chi2(){
  mus_chi2_isLoaded = true;
}
void HWW::Load_mus_ndof(){
  mus_ndof_isLoaded = true;
}
void HWW::Load_mus_ip3d(){
  mus_ip3d_isLoaded = true;
}
void HWW::Load_mus_ip3derr(){
  mus_ip3derr_isLoaded = true;
}
void HWW::Load_mus_segmCompatibility(){
  mus_segmCompatibility_isLoaded = true;
}
void HWW::Load_mus_caloCompatibility(){
  mus_caloCompatibility_isLoaded = true;
}
void HWW::Load_mus_e_had(){
  mus_e_had_isLoaded = true;
}
void HWW::Load_mus_e_ho(){
  mus_e_ho_isLoaded = true;
}
void HWW::Load_mus_e_em(){
  mus_e_em_isLoaded = true;
}
void HWW::Load_mus_e_hadS9(){
  mus_e_hadS9_isLoaded = true;
}
void HWW::Load_mus_e_hoS9(){
  mus_e_hoS9_isLoaded = true;
}
void HWW::Load_mus_e_emS9(){
  mus_e_emS9_isLoaded = true;
}
void HWW::Load_mus_iso03_sumPt(){
  mus_iso03_sumPt_isLoaded = true;
}
void HWW::Load_mus_iso03_emEt(){
  mus_iso03_emEt_isLoaded = true;
}
void HWW::Load_mus_iso03_hadEt(){
  mus_iso03_hadEt_isLoaded = true;
}
void HWW::Load_mus_iso05_sumPt(){
  mus_iso05_sumPt_isLoaded = true;
}
void HWW::Load_mus_iso05_emEt(){
  mus_iso05_emEt_isLoaded = true;
}
void HWW::Load_mus_iso05_hadEt(){
  mus_iso05_hadEt_isLoaded = true;
}
void HWW::Load_mus_sta_d0(){
  mus_sta_d0_isLoaded = true;
}
void HWW::Load_mus_sta_z0corr(){
  mus_sta_z0corr_isLoaded = true;
}
void HWW::Load_mus_isoR03_pf_ChargedHadronPt(){
  mus_isoR03_pf_ChargedHadronPt_isLoaded = true;
}
void HWW::Load_mus_isoR03_pf_NeutralHadronEt(){
  mus_isoR03_pf_NeutralHadronEt_isLoaded = true;
}
void HWW::Load_mus_isoR03_pf_PhotonEt(){
  mus_isoR03_pf_PhotonEt_isLoaded = true;
}
void HWW::Load_mus_isoR03_pf_PUPt(){
  mus_isoR03_pf_PUPt_isLoaded = true;
}
void HWW::Load_mus_iso_ecalvetoDep(){
  mus_iso_ecalvetoDep_isLoaded = true;
}
void HWW::Load_mus_iso_hcalvetoDep(){
  mus_iso_hcalvetoDep_isLoaded = true;
}
void HWW::Load_mus_gfit_validSTAHits(){
  mus_gfit_validSTAHits_isLoaded = true;
}
void HWW::Load_mus_numberOfMatchedStations(){
  mus_numberOfMatchedStations_isLoaded = true;
}
void HWW::Load_mus_pfmusidx(){
  mus_pfmusidx_isLoaded = true;
}
void HWW::Load_mus_charge(){
  mus_charge_isLoaded = true;
}
void HWW::Load_mus_validHits(){
  mus_validHits_isLoaded = true;
}
void HWW::Load_mus_trkidx(){
  mus_trkidx_isLoaded = true;
}
void HWW::Load_mus_pid_PFMuon(){
  mus_pid_PFMuon_isLoaded = true;
}
void HWW::Load_mus_pid_TMLastStationTight(){
  mus_pid_TMLastStationTight_isLoaded = true;
}
void HWW::Load_mus_nmatches(){
  mus_nmatches_isLoaded = true;
}
void HWW::Load_mus_goodmask(){
  mus_goodmask_isLoaded = true;
}
void HWW::Load_mus_type(){
  mus_type_isLoaded = true;
}
void HWW::Load_hyp_jets_p4(){
  hyp_jets_p4_isLoaded = true;
}
void HWW::Load_hyp_p4(){
  hyp_p4_isLoaded = true;
}
void HWW::Load_hyp_ll_p4(){
  hyp_ll_p4_isLoaded = true;
}
void HWW::Load_hyp_lt_p4(){
  hyp_lt_p4_isLoaded = true;
}
void HWW::Load_hyp_ll_index(){
  hyp_ll_index_isLoaded = true;
}
void HWW::Load_hyp_lt_index(){
  hyp_lt_index_isLoaded = true;
}
void HWW::Load_hyp_ll_id(){
  hyp_ll_id_isLoaded = true;
}
void HWW::Load_hyp_lt_id(){
  hyp_lt_id_isLoaded = true;
}
void HWW::Load_hyp_ll_charge(){
  hyp_ll_charge_isLoaded = true;
}
void HWW::Load_hyp_lt_charge(){
  hyp_lt_charge_isLoaded = true;
}
void HWW::Load_hyp_type(){
  hyp_type_isLoaded = true;
}
void HWW::Load_evt_run(){
  evt_run_isLoaded = true;
}
void HWW::Load_evt_lumiBlock(){
  evt_lumiBlock_isLoaded = true;
}
void HWW::Load_evt_event(){
  evt_event_isLoaded = true;
}
void HWW::Load_evt_isRealData(){
  evt_isRealData_isLoaded = true;
}
void HWW::Load_evt_ww_rho_vor(){
  evt_ww_rho_vor_isLoaded = true;
}
void HWW::Load_evt_ww_rho(){
  evt_ww_rho_isLoaded = true;
}
void HWW::Load_evt_rho(){
  evt_rho_isLoaded = true;
}
void HWW::Load_evt_kt6pf_foregiso_rho(){
  evt_kt6pf_foregiso_rho_isLoaded = true;
}
void HWW::Load_evt_pfmet(){
  evt_pfmet_isLoaded = true;
}
void HWW::Load_evt_pfmetPhi(){
  evt_pfmetPhi_isLoaded = true;
}
void HWW::Load_convs_ndof(){
  convs_ndof_isLoaded = true;
}
void HWW::Load_convs_chi2(){
  convs_chi2_isLoaded = true;
}
void HWW::Load_convs_dl(){
  convs_dl_isLoaded = true;
}
void HWW::Load_convs_isConverted(){
  convs_isConverted_isLoaded = true;
}
void HWW::Load_convs_tkalgo(){
  convs_tkalgo_isLoaded = true;
}
void HWW::Load_convs_tkidx(){
  convs_tkidx_isLoaded = true;
}
void HWW::Load_convs_nHitsBeforeVtx(){
  convs_nHitsBeforeVtx_isLoaded = true;
}
void HWW::Load_convs_quality(){
  convs_quality_isLoaded = true;
}
void HWW::Load_scs_sigmaIEtaIPhi(){
  scs_sigmaIEtaIPhi_isLoaded = true;
}
void HWW::Load_scs_pos_p4(){
  scs_pos_p4_isLoaded = true;
}
void HWW::Load_gsftrks_p4(){
  gsftrks_p4_isLoaded = true;
}
void HWW::Load_gsftrks_vertex_p4(){
  gsftrks_vertex_p4_isLoaded = true;
}
void HWW::Load_gsftrks_d0(){
  gsftrks_d0_isLoaded = true;
}
void HWW::Load_gsftrks_d0Err(){
  gsftrks_d0Err_isLoaded = true;
}
void HWW::Load_gsftrks_phiErr(){
  gsftrks_phiErr_isLoaded = true;
}
void HWW::Load_gsftrks_d0phiCov(){
  gsftrks_d0phiCov_isLoaded = true;
}
void HWW::Load_gsftrks_z0Err(){
  gsftrks_z0Err_isLoaded = true;
}
void HWW::Load_gsftrks_z0(){
  gsftrks_z0_isLoaded = true;
}
void HWW::Load_gsftrks_etaErr(){
  gsftrks_etaErr_isLoaded = true;
}
void HWW::Load_pfcands_p4(){
  pfcands_p4_isLoaded = true;
}
void HWW::Load_pfcands_trkidx(){
  pfcands_trkidx_isLoaded = true;
}
void HWW::Load_pfcands_particleId(){
  pfcands_particleId_isLoaded = true;
}
void HWW::Load_pfcands_pfelsidx(){
  pfcands_pfelsidx_isLoaded = true;
}
void HWW::Load_pfcands_vtxidx(){
  pfcands_vtxidx_isLoaded = true;
}
void HWW::Load_pfcands_charge(){
  pfcands_charge_isLoaded = true;
}
void HWW::Load_pfels_elsidx(){
  pfels_elsidx_isLoaded = true;
}
void HWW::Load_pfels_p4(){
  pfels_p4_isLoaded = true;
}
void HWW::Load_pfmus_p4(){
  pfmus_p4_isLoaded = true;
}
void HWW::Load_trk_met(){
  trk_met_isLoaded = true;
}
void HWW::Load_trk_metPhi(){
  trk_metPhi_isLoaded = true;
}
void HWW::Load_pfjets_p4(){
  pfjets_p4_isLoaded = true;
}
void HWW::Load_pfjets_corr_p4(){
  pfjets_corr_p4_isLoaded = true;
}
void HWW::Load_pfjets_area(){
  pfjets_area_isLoaded = true;
}
void HWW::Load_pfjets_JEC(){
  pfjets_JEC_isLoaded = true;
}
void HWW::Load_pfjets_mvavalue(){
  pfjets_mvavalue_isLoaded = true;
}
void HWW::Load_pfjets_trackCountingHighEffBJetTag(){
  pfjets_trackCountingHighEffBJetTag_isLoaded = true;
}


HWW::HWW(){

  vtxs_position_.clear();
  vtxs_ndof_.clear();
  vtxs_sumpt_.clear();
  vtxs_isFake_.clear();
  vtxs_xError_.clear();
  vtxs_yError_.clear();
  vtxs_zError_.clear();
  vtxs_covMatrix_.clear();

  
  trks_trk_p4_.clear();
  trks_vertex_p4_.clear();
  trks_chi2_.clear();
  trks_ndof_.clear();
  trks_d0_.clear();
  trks_nlayers_.clear();
  trks_valid_pixelhits_.clear();
  trks_z0_.clear();
  trks_z0Err_.clear();
  trks_etaErr_.clear();
  trks_d0Err_.clear();
  trks_phiErr_.clear();
  trks_d0phiCov_.clear();
  trks_qualityMask_.clear();
  trks_charge_.clear();

  
  els_p4_.clear();
  els_trk_p4_.clear();
  els_vertex_p4_.clear();
  els_lh_.clear();
  els_etaSC_.clear();
  els_sigmaIEtaIEta_.clear();
  els_dEtaIn_.clear();
  els_dPhiIn_.clear();
  els_hOverE_.clear();
  els_tkIso_.clear();
  els_d0corr_.clear();
  els_d0_.clear();
  els_z0corr_.clear();
  els_fbrem_.clear();
  els_eOverPIn_.clear();
  els_eSeedOverPOut_.clear();
  els_eSeedOverPIn_.clear();
  els_sigmaIPhiIPhi_.clear();
  els_eSC_.clear();
  els_ip3d_.clear();
  els_ip3derr_.clear();
  els_chi2_.clear();
  els_ndof_.clear();
  els_dEtaOut_.clear();
  els_dPhiOut_.clear();
  els_eSCRaw_.clear();
  els_etaSCwidth_.clear();
  els_phiSCwidth_.clear();
  els_eSCPresh_.clear();
  els_iso03_pf_ch_.clear();
  els_iso03_pf_nhad05_.clear();
  els_iso03_pf_gamma05_.clear();
  els_iso04_pf_ch_.clear();
  els_iso04_pf_nhad05_.clear();
  els_iso04_pf_gamma05_.clear();
  els_e5x5_.clear();
  els_e1x5_.clear();
  els_e3x3_.clear();
  els_ecalEnergy_.clear();
  els_eOverPOut_.clear();
  els_ecalIso_.clear();
  els_hcalIso_.clear();
  els_trkshFrac_.clear();
  els_conv_dist_.clear();
  els_conv_dcot_.clear();
  els_conv_old_dist_.clear();
  els_conv_old_dcot_.clear();
  els_iso04_pf2012_ch_.clear();
  els_iso04_pf2012_em_.clear();
  els_iso04_pf2012_nh_.clear();
  els_iso03_pf2012_ch_.clear();
  els_iso03_pf2012_em_.clear();
  els_iso03_pf2012_nh_.clear();
  els_ecalIso04_.clear();
  els_hcalIso04_.clear();
  els_nSeed_.clear();
  els_scindex_.clear();
  els_charge_.clear();
  els_gsftrkidx_.clear();
  els_exp_innerlayers_.clear();
  els_trkidx_.clear();
  els_type_.clear();
  els_fiduciality_.clear();
  els_sccharge_.clear();
  els_trk_charge_.clear();
  els_closestMuon_.clear();

  
  mus_p4_.clear();
  mus_trk_p4_.clear();
  mus_vertex_p4_.clear();
  mus_sta_p4_.clear();
  mus_gfit_chi2_.clear();
  mus_gfit_ndof_.clear();
  mus_ptErr_.clear();
  mus_trkKink_.clear();
  mus_d0corr_.clear();
  mus_d0_.clear();
  mus_z0corr_.clear();
  mus_chi2_.clear();
  mus_ndof_.clear();
  mus_ip3d_.clear();
  mus_ip3derr_.clear();
  mus_segmCompatibility_.clear();
  mus_caloCompatibility_.clear();
  mus_e_had_.clear();
  mus_e_ho_.clear();
  mus_e_em_.clear();
  mus_e_hadS9_.clear();
  mus_e_hoS9_.clear();
  mus_e_emS9_.clear();
  mus_iso03_sumPt_.clear();
  mus_iso03_emEt_.clear();
  mus_iso03_hadEt_.clear();
  mus_iso05_sumPt_.clear();
  mus_iso05_emEt_.clear();
  mus_iso05_hadEt_.clear();
  mus_sta_d0_.clear();
  mus_sta_z0corr_.clear();
  mus_isoR03_pf_ChargedHadronPt_.clear();
  mus_isoR03_pf_NeutralHadronEt_.clear();
  mus_isoR03_pf_PhotonEt_.clear();
  mus_isoR03_pf_PUPt_.clear();
  mus_iso_ecalvetoDep_.clear();
  mus_iso_hcalvetoDep_.clear();
  mus_gfit_validSTAHits_.clear();
  mus_numberOfMatchedStations_.clear();
  mus_pfmusidx_.clear();
  mus_charge_.clear();
  mus_validHits_.clear();
  mus_trkidx_.clear();
  mus_pid_PFMuon_.clear();
  mus_pid_TMLastStationTight_.clear();
  mus_nmatches_.clear();
  mus_goodmask_.clear();
  mus_type_.clear();

  
  hyp_jets_p4_.clear();
  hyp_p4_.clear();
  hyp_ll_p4_.clear();
  hyp_lt_p4_.clear();
  hyp_ll_index_.clear();
  hyp_lt_index_.clear();
  hyp_ll_id_.clear();
  hyp_lt_id_.clear();
  hyp_ll_charge_.clear();
  hyp_lt_charge_.clear();
  hyp_type_.clear();

  
  evt_run_ = 999;
  evt_lumiBlock_ = 999;
  evt_event_ = 999;
  evt_isRealData_ = -999;
  evt_ww_rho_vor_ = -999.0;
  evt_ww_rho_ = -999.0;
  evt_rho_ = -999.0;
  evt_kt6pf_foregiso_rho_ = -999.0;
  evt_pfmet_ = -999.0;
  evt_pfmetPhi_ = -999.0;

  convs_ndof_.clear();
  convs_chi2_.clear();
  convs_dl_.clear();
  convs_isConverted_.clear();
  convs_tkalgo_.clear();
  convs_tkidx_.clear();
  convs_nHitsBeforeVtx_.clear();
  convs_quality_.clear();
  scs_sigmaIEtaIPhi_.clear();
  scs_pos_p4_.clear();
  gsftrks_p4_.clear();
  gsftrks_vertex_p4_.clear();
  gsftrks_d0_.clear();
  gsftrks_d0Err_.clear();
  gsftrks_phiErr_.clear();
  gsftrks_d0phiCov_.clear();
  gsftrks_z0Err_.clear();
  gsftrks_z0_.clear();
  gsftrks_etaErr_.clear();
  pfcands_p4_.clear();
  pfcands_trkidx_.clear();
  pfcands_particleId_.clear();
  pfcands_pfelsidx_.clear();
  pfcands_vtxidx_.clear();
  pfcands_charge_.clear();
  pfels_elsidx_.clear();
  pfels_p4_.clear();
  pfmus_p4_.clear();
  trk_met_.clear();
  trk_metPhi_.clear();
  pfjets_p4_.clear();
  pfjets_corr_p4_.clear();
  pfjets_area_.clear();
  pfjets_JEC_.clear();
  pfjets_mvavalue_.clear();
  pfjets_trackCountingHighEffBJetTag_.clear();



  vtxs_position_isLoaded = false;
  vtxs_ndof_isLoaded = false;
  vtxs_sumpt_isLoaded = false;
  vtxs_isFake_isLoaded = false;
  vtxs_xError_isLoaded = false;
  vtxs_yError_isLoaded = false;
  vtxs_zError_isLoaded = false;
  vtxs_covMatrix_isLoaded = false;

  trks_trk_p4_isLoaded = false;
  trks_vertex_p4_isLoaded = false;
  trks_chi2_isLoaded = false;
  trks_ndof_isLoaded = false;
  trks_d0_isLoaded = false;
  trks_nlayers_isLoaded = false;
  trks_valid_pixelhits_isLoaded = false;
  trks_z0_isLoaded = false;
  trks_z0Err_isLoaded = false;
  trks_etaErr_isLoaded = false;
  trks_d0Err_isLoaded = false;
  trks_phiErr_isLoaded = false;
  trks_d0phiCov_isLoaded = false;
  trks_qualityMask_isLoaded = false;
  trks_charge_isLoaded = false;

  els_p4_isLoaded = false;
  els_trk_p4_isLoaded = false;
  els_vertex_p4_isLoaded = false;
  els_lh_isLoaded = false;
  els_etaSC_isLoaded = false;
  els_sigmaIEtaIEta_isLoaded = false;
  els_dEtaIn_isLoaded = false;
  els_dPhiIn_isLoaded = false;
  els_hOverE_isLoaded = false;
  els_tkIso_isLoaded = false;
  els_d0corr_isLoaded = false;
  els_d0_isLoaded = false;
  els_z0corr_isLoaded = false;
  els_fbrem_isLoaded = false;
  els_eOverPIn_isLoaded = false;
  els_eSeedOverPOut_isLoaded = false;
  els_eSeedOverPIn_isLoaded = false;
  els_sigmaIPhiIPhi_isLoaded = false;
  els_eSC_isLoaded = false;
  els_ip3d_isLoaded = false;
  els_ip3derr_isLoaded = false;
  els_chi2_isLoaded = false;
  els_ndof_isLoaded = false;
  els_dEtaOut_isLoaded = false;
  els_dPhiOut_isLoaded = false;
  els_eSCRaw_isLoaded = false;
  els_etaSCwidth_isLoaded = false;
  els_phiSCwidth_isLoaded = false;
  els_eSCPresh_isLoaded = false;
  els_iso03_pf_ch_isLoaded = false;
  els_iso03_pf_nhad05_isLoaded = false;
  els_iso03_pf_gamma05_isLoaded = false;
  els_iso04_pf_ch_isLoaded = false;
  els_iso04_pf_nhad05_isLoaded = false;
  els_iso04_pf_gamma05_isLoaded = false;
  els_e5x5_isLoaded = false;
  els_e1x5_isLoaded = false;
  els_e3x3_isLoaded = false;
  els_ecalEnergy_isLoaded = false;
  els_eOverPOut_isLoaded = false;
  els_ecalIso_isLoaded = false;
  els_hcalIso_isLoaded = false;
  els_trkshFrac_isLoaded = false;
  els_conv_dist_isLoaded = false;
  els_conv_dcot_isLoaded = false;
  els_conv_old_dist_isLoaded = false;
  els_conv_old_dcot_isLoaded = false;
  els_iso04_pf2012_ch_isLoaded = false;
  els_iso04_pf2012_em_isLoaded = false;
  els_iso04_pf2012_nh_isLoaded = false;
  els_iso03_pf2012_ch_isLoaded = false;
  els_iso03_pf2012_em_isLoaded = false;
  els_iso03_pf2012_nh_isLoaded = false;
  els_ecalIso04_isLoaded = false;
  els_hcalIso04_isLoaded = false;
  els_nSeed_isLoaded = false;
  els_scindex_isLoaded = false;
  els_charge_isLoaded = false;
  els_gsftrkidx_isLoaded = false;
  els_exp_innerlayers_isLoaded = false;
  els_trkidx_isLoaded = false;
  els_type_isLoaded = false;
  els_fiduciality_isLoaded = false;
  els_sccharge_isLoaded = false;
  els_trk_charge_isLoaded = false;
  els_closestMuon_isLoaded = false;

  mus_p4_isLoaded = false;
  mus_trk_p4_isLoaded = false;
  mus_vertex_p4_isLoaded = false;
  mus_sta_p4_isLoaded = false;
  mus_gfit_chi2_isLoaded = false;
  mus_gfit_ndof_isLoaded = false;
  mus_ptErr_isLoaded = false;
  mus_trkKink_isLoaded = false;
  mus_d0corr_isLoaded = false;
  mus_d0_isLoaded = false;
  mus_z0corr_isLoaded = false;
  mus_chi2_isLoaded = false;
  mus_ndof_isLoaded = false;
  mus_ip3d_isLoaded = false;
  mus_ip3derr_isLoaded = false;
  mus_segmCompatibility_isLoaded = false;
  mus_caloCompatibility_isLoaded = false;
  mus_e_had_isLoaded = false;
  mus_e_ho_isLoaded = false;
  mus_e_em_isLoaded = false;
  mus_e_hadS9_isLoaded = false;
  mus_e_hoS9_isLoaded = false;
  mus_e_emS9_isLoaded = false;
  mus_iso03_sumPt_isLoaded = false;
  mus_iso03_emEt_isLoaded = false;
  mus_iso03_hadEt_isLoaded = false;
  mus_iso05_sumPt_isLoaded = false;
  mus_iso05_emEt_isLoaded = false;
  mus_iso05_hadEt_isLoaded = false;
  mus_sta_d0_isLoaded = false;
  mus_sta_z0corr_isLoaded = false;
  mus_isoR03_pf_ChargedHadronPt_isLoaded = false;
  mus_isoR03_pf_NeutralHadronEt_isLoaded = false;
  mus_isoR03_pf_PhotonEt_isLoaded = false;
  mus_isoR03_pf_PUPt_isLoaded = false;
  mus_iso_ecalvetoDep_isLoaded = false;
  mus_iso_hcalvetoDep_isLoaded = false;
  mus_gfit_validSTAHits_isLoaded = false;
  mus_numberOfMatchedStations_isLoaded = false;
  mus_pfmusidx_isLoaded = false;
  mus_charge_isLoaded = false;
  mus_validHits_isLoaded = false;
  mus_trkidx_isLoaded = false;
  mus_pid_PFMuon_isLoaded = false;
  mus_pid_TMLastStationTight_isLoaded = false;
  mus_nmatches_isLoaded = false;
  mus_goodmask_isLoaded = false;
  mus_type_isLoaded = false;

  hyp_jets_p4_isLoaded = false;
  hyp_p4_isLoaded = false;
  hyp_ll_p4_isLoaded = false;
  hyp_lt_p4_isLoaded = false;
  hyp_ll_index_isLoaded = false;
  hyp_lt_index_isLoaded = false;
  hyp_ll_id_isLoaded = false;
  hyp_lt_id_isLoaded = false;
  hyp_ll_charge_isLoaded = false;
  hyp_lt_charge_isLoaded = false;
  hyp_type_isLoaded = false;

  evt_run_isLoaded = false;
  evt_lumiBlock_isLoaded = false;
  evt_event_isLoaded = false;
  evt_isRealData_isLoaded = false;
  evt_ww_rho_vor_isLoaded = false;
  evt_ww_rho_isLoaded = false;
  evt_rho_isLoaded = false;
  evt_kt6pf_foregiso_rho_isLoaded = false;
  evt_pfmet_isLoaded = false;
  evt_pfmetPhi_isLoaded = false;

  convs_ndof_isLoaded = false;
  convs_chi2_isLoaded = false;
  convs_dl_isLoaded = false;
  convs_isConverted_isLoaded = false;
  convs_tkalgo_isLoaded = false;
  convs_tkidx_isLoaded = false;
  convs_nHitsBeforeVtx_isLoaded = false;
  convs_quality_isLoaded = false;
  scs_sigmaIEtaIPhi_isLoaded = false;
  scs_pos_p4_isLoaded = false;
  gsftrks_p4_isLoaded = false;
  gsftrks_vertex_p4_isLoaded = false;
  gsftrks_d0_isLoaded = false;
  gsftrks_d0Err_isLoaded = false;
  gsftrks_phiErr_isLoaded = false;
  gsftrks_d0phiCov_isLoaded = false;
  gsftrks_z0Err_isLoaded = false;
  gsftrks_z0_isLoaded = false;
  gsftrks_etaErr_isLoaded = false;
  pfcands_p4_isLoaded = false;
  pfcands_trkidx_isLoaded = false;
  pfcands_particleId_isLoaded = false;
  pfcands_pfelsidx_isLoaded = false;
  pfcands_vtxidx_isLoaded = false;
  pfcands_charge_isLoaded = false;
  pfels_elsidx_isLoaded = false;
  pfels_p4_isLoaded = false;
  pfmus_p4_isLoaded = false;
  trk_met_isLoaded = false;
  trk_metPhi_isLoaded = false;
  pfjets_p4_isLoaded = false;
  pfjets_corr_p4_isLoaded = false;
  pfjets_area_isLoaded = false;
  pfjets_JEC_isLoaded = false;
  pfjets_mvavalue_isLoaded = false;
  pfjets_trackCountingHighEffBJetTag_isLoaded = false;

}
