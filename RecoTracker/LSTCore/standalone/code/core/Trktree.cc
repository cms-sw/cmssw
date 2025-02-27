#include "Trktree.h"
Trktree trk;

void Trktree::Init(TTree *tree) {
  tree->SetMakeClass(1);

  // Added by Kasia --------------------------------------------------------------------
  sim_etadiffs_branch = 0;
  if (tree->GetBranch("sim_etadiffs") != 0) {
    sim_etadiffs_branch = tree->GetBranch("sim_etadiffs");
    if (sim_etadiffs_branch) { sim_etadiffs_branch->SetAddress(&sim_etadiffs_); }
  }
  sim_phidiffs_branch = 0;
  if (tree->GetBranch("sim_phidiffs") != 0) {
    sim_phidiffs_branch = tree->GetBranch("sim_phidiffs");
    if (sim_phidiffs_branch) { sim_phidiffs_branch->SetAddress(&sim_phidiffs_); }
  }
  sim_rjet_branch = 0;
  if (tree->GetBranch("sim_rjet") != 0) {
    sim_rjet_branch = tree->GetBranch("sim_rjet");
    if (sim_rjet_branch) { sim_rjet_branch->SetAddress(&sim_rjet_); }
  }
  sim_jet_eta_branch = 0;
  if (tree->GetBranch("sim_jet_eta") != 0) {
    sim_jet_eta_branch = tree->GetBranch("sim_jet_eta");
    if (sim_jet_eta_branch) { sim_jet_eta_branch->SetAddress(&sim_jet_eta_); }
  }
  sim_jet_phi_branch = 0;
  if (tree->GetBranch("sim_jet_phi") != 0) {
    sim_jet_phi_branch = tree->GetBranch("sim_jet_phi");
    if (sim_jet_phi_branch) { sim_jet_phi_branch->SetAddress(&sim_jet_phi_); }
  }
  sim_jet_pt_branch = 0;
  if (tree->GetBranch("sim_jet_pt") != 0) {
    sim_jet_pt_branch = tree->GetBranch("sim_jet_pt");
    if (sim_jet_pt_branch) { sim_jet_pt_branch->SetAddress(&sim_jet_pt_); }
  }
  //------------------------------------------------------------------------------------

  see_stateCcov01_branch = 0;
  if (tree->GetBranch("see_stateCcov01") != 0) {
    see_stateCcov01_branch = tree->GetBranch("see_stateCcov01");
    if (see_stateCcov01_branch) {
      see_stateCcov01_branch->SetAddress(&see_stateCcov01_);
    }
  }
  simhit_rod_branch = 0;
  if (tree->GetBranch("simhit_rod") != 0) {
    simhit_rod_branch = tree->GetBranch("simhit_rod");
    if (simhit_rod_branch) {
      simhit_rod_branch->SetAddress(&simhit_rod_);
    }
  }
  trk_phi_branch = 0;
  if (tree->GetBranch("trk_phi") != 0) {
    trk_phi_branch = tree->GetBranch("trk_phi");
    if (trk_phi_branch) {
      trk_phi_branch->SetAddress(&trk_phi_);
    }
  }
  bsp_x_branch = 0;
  if (tree->GetBranch("bsp_x") != 0) {
    bsp_x_branch = tree->GetBranch("bsp_x");
    if (bsp_x_branch) {
      bsp_x_branch->SetAddress(&bsp_x_);
    }
  }
  see_stateCcov05_branch = 0;
  if (tree->GetBranch("see_stateCcov05") != 0) {
    see_stateCcov05_branch = tree->GetBranch("see_stateCcov05");
    if (see_stateCcov05_branch) {
      see_stateCcov05_branch->SetAddress(&see_stateCcov05_);
    }
  }
  see_stateCcov04_branch = 0;
  if (tree->GetBranch("see_stateCcov04") != 0) {
    see_stateCcov04_branch = tree->GetBranch("see_stateCcov04");
    if (see_stateCcov04_branch) {
      see_stateCcov04_branch->SetAddress(&see_stateCcov04_);
    }
  }
  trk_dxyPV_branch = 0;
  if (tree->GetBranch("trk_dxyPV") != 0) {
    trk_dxyPV_branch = tree->GetBranch("trk_dxyPV");
    if (trk_dxyPV_branch) {
      trk_dxyPV_branch->SetAddress(&trk_dxyPV_);
    }
  }
  simhit_tof_branch = 0;
  if (tree->GetBranch("simhit_tof") != 0) {
    simhit_tof_branch = tree->GetBranch("simhit_tof");
    if (simhit_tof_branch) {
      simhit_tof_branch->SetAddress(&simhit_tof_);
    }
  }
  sim_event_branch = 0;
  if (tree->GetBranch("sim_event") != 0) {
    sim_event_branch = tree->GetBranch("sim_event");
    if (sim_event_branch) {
      sim_event_branch->SetAddress(&sim_event_);
    }
  }
  simhit_isStack_branch = 0;
  if (tree->GetBranch("simhit_isStack") != 0) {
    simhit_isStack_branch = tree->GetBranch("simhit_isStack");
    if (simhit_isStack_branch) {
      simhit_isStack_branch->SetAddress(&simhit_isStack_);
    }
  }
  trk_dz_branch = 0;
  if (tree->GetBranch("trk_dz") != 0) {
    trk_dz_branch = tree->GetBranch("trk_dz");
    if (trk_dz_branch) {
      trk_dz_branch->SetAddress(&trk_dz_);
    }
  }
  see_stateCcov03_branch = 0;
  if (tree->GetBranch("see_stateCcov03") != 0) {
    see_stateCcov03_branch = tree->GetBranch("see_stateCcov03");
    if (see_stateCcov03_branch) {
      see_stateCcov03_branch->SetAddress(&see_stateCcov03_);
    }
  }
  sim_eta_branch = 0;
  if (tree->GetBranch("sim_eta") != 0) {
    sim_eta_branch = tree->GetBranch("sim_eta");
    if (sim_eta_branch) {
      sim_eta_branch->SetAddress(&sim_eta_);
    }
  }
  simvtx_processType_branch = 0;
  if (tree->GetBranch("simvtx_processType") != 0) {
    simvtx_processType_branch = tree->GetBranch("simvtx_processType");
    if (simvtx_processType_branch) {
      simvtx_processType_branch->SetAddress(&simvtx_processType_);
    }
  }
  pix_radL_branch = 0;
  if (tree->GetBranch("pix_radL") != 0) {
    pix_radL_branch = tree->GetBranch("pix_radL");
    if (pix_radL_branch) {
      pix_radL_branch->SetAddress(&pix_radL_);
    }
  }
  see_stateCcov02_branch = 0;
  if (tree->GetBranch("see_stateCcov02") != 0) {
    see_stateCcov02_branch = tree->GetBranch("see_stateCcov02");
    if (see_stateCcov02_branch) {
      see_stateCcov02_branch->SetAddress(&see_stateCcov02_);
    }
  }
  see_nGlued_branch = 0;
  if (tree->GetBranch("see_nGlued") != 0) {
    see_nGlued_branch = tree->GetBranch("see_nGlued");
    if (see_nGlued_branch) {
      see_nGlued_branch->SetAddress(&see_nGlued_);
    }
  }
  trk_bestSimTrkIdx_branch = 0;
  if (tree->GetBranch("trk_bestSimTrkIdx") != 0) {
    trk_bestSimTrkIdx_branch = tree->GetBranch("trk_bestSimTrkIdx");
    if (trk_bestSimTrkIdx_branch) {
      trk_bestSimTrkIdx_branch->SetAddress(&trk_bestSimTrkIdx_);
    }
  }
  see_stateTrajGlbPz_branch = 0;
  if (tree->GetBranch("see_stateTrajGlbPz") != 0) {
    see_stateTrajGlbPz_branch = tree->GetBranch("see_stateTrajGlbPz");
    if (see_stateTrajGlbPz_branch) {
      see_stateTrajGlbPz_branch->SetAddress(&see_stateTrajGlbPz_);
    }
  }
  pix_yz_branch = 0;
  if (tree->GetBranch("pix_yz") != 0) {
    pix_yz_branch = tree->GetBranch("pix_yz");
    if (pix_yz_branch) {
      pix_yz_branch->SetAddress(&pix_yz_);
    }
  }
  pix_yy_branch = 0;
  if (tree->GetBranch("pix_yy") != 0) {
    pix_yy_branch = tree->GetBranch("pix_yy");
    if (pix_yy_branch) {
      pix_yy_branch->SetAddress(&pix_yy_);
    }
  }
  simhit_process_branch = 0;
  if (tree->GetBranch("simhit_process") != 0) {
    simhit_process_branch = tree->GetBranch("simhit_process");
    if (simhit_process_branch) {
      simhit_process_branch->SetAddress(&simhit_process_);
    }
  }
  see_stateCcov34_branch = 0;
  if (tree->GetBranch("see_stateCcov34") != 0) {
    see_stateCcov34_branch = tree->GetBranch("see_stateCcov34");
    if (see_stateCcov34_branch) {
      see_stateCcov34_branch->SetAddress(&see_stateCcov34_);
    }
  }
  trk_nInnerLost_branch = 0;
  if (tree->GetBranch("trk_nInnerLost") != 0) {
    trk_nInnerLost_branch = tree->GetBranch("trk_nInnerLost");
    if (trk_nInnerLost_branch) {
      trk_nInnerLost_branch->SetAddress(&trk_nInnerLost_);
    }
  }
  see_py_branch = 0;
  if (tree->GetBranch("see_py") != 0) {
    see_py_branch = tree->GetBranch("see_py");
    if (see_py_branch) {
      see_py_branch->SetAddress(&see_py_);
    }
  }
  sim_trkIdx_branch = 0;
  if (tree->GetBranch("sim_trkIdx") != 0) {
    sim_trkIdx_branch = tree->GetBranch("sim_trkIdx");
    if (sim_trkIdx_branch) {
      sim_trkIdx_branch->SetAddress(&sim_trkIdx_);
    }
  }
  trk_nLost_branch = 0;
  if (tree->GetBranch("trk_nLost") != 0) {
    trk_nLost_branch = tree->GetBranch("trk_nLost");
    if (trk_nLost_branch) {
      trk_nLost_branch->SetAddress(&trk_nLost_);
    }
  }
  pix_isBarrel_branch = 0;
  if (tree->GetBranch("pix_isBarrel") != 0) {
    pix_isBarrel_branch = tree->GetBranch("pix_isBarrel");
    if (pix_isBarrel_branch) {
      pix_isBarrel_branch->SetAddress(&pix_isBarrel_);
    }
  }
  see_dxyErr_branch = 0;
  if (tree->GetBranch("see_dxyErr") != 0) {
    see_dxyErr_branch = tree->GetBranch("see_dxyErr");
    if (see_dxyErr_branch) {
      see_dxyErr_branch->SetAddress(&see_dxyErr_);
    }
  }
  simhit_detId_branch = 0;
  if (tree->GetBranch("simhit_detId") != 0) {
    simhit_detId_branch = tree->GetBranch("simhit_detId");
    if (simhit_detId_branch) {
      simhit_detId_branch->SetAddress(&simhit_detId_);
    }
  }
  simhit_subdet_branch = 0;
  if (tree->GetBranch("simhit_subdet") != 0) {
    simhit_subdet_branch = tree->GetBranch("simhit_subdet");
    if (simhit_subdet_branch) {
      simhit_subdet_branch->SetAddress(&simhit_subdet_);
    }
  }
  see_hitIdx_branch = 0;
  if (tree->GetBranch("see_hitIdx") != 0) {
    see_hitIdx_branch = tree->GetBranch("see_hitIdx");
    if (see_hitIdx_branch) {
      see_hitIdx_branch->SetAddress(&see_hitIdx_);
    }
  }
  see_pt_branch = 0;
  if (tree->GetBranch("see_pt") != 0) {
    see_pt_branch = tree->GetBranch("see_pt");
    if (see_pt_branch) {
      see_pt_branch->SetAddress(&see_pt_);
    }
  }
  ph2_detId_branch = 0;
  if (tree->GetBranch("ph2_detId") != 0) {
    ph2_detId_branch = tree->GetBranch("ph2_detId");
    if (ph2_detId_branch) {
      ph2_detId_branch->SetAddress(&ph2_detId_);
    }
  }
  trk_nStripLay_branch = 0;
  if (tree->GetBranch("trk_nStripLay") != 0) {
    trk_nStripLay_branch = tree->GetBranch("trk_nStripLay");
    if (trk_nStripLay_branch) {
      trk_nStripLay_branch->SetAddress(&trk_nStripLay_);
    }
  }
  see_bestFromFirstHitSimTrkIdx_branch = 0;
  if (tree->GetBranch("see_bestFromFirstHitSimTrkIdx") != 0) {
    see_bestFromFirstHitSimTrkIdx_branch = tree->GetBranch("see_bestFromFirstHitSimTrkIdx");
    if (see_bestFromFirstHitSimTrkIdx_branch) {
      see_bestFromFirstHitSimTrkIdx_branch->SetAddress(&see_bestFromFirstHitSimTrkIdx_);
    }
  }
  sim_pca_pt_branch = 0;
  if (tree->GetBranch("sim_pca_pt") != 0) {
    sim_pca_pt_branch = tree->GetBranch("sim_pca_pt");
    if (sim_pca_pt_branch) {
      sim_pca_pt_branch->SetAddress(&sim_pca_pt_);
    }
  }
  see_trkIdx_branch = 0;
  if (tree->GetBranch("see_trkIdx") != 0) {
    see_trkIdx_branch = tree->GetBranch("see_trkIdx");
    if (see_trkIdx_branch) {
      see_trkIdx_branch->SetAddress(&see_trkIdx_);
    }
  }
  trk_nCluster_branch = 0;
  if (tree->GetBranch("trk_nCluster") != 0) {
    trk_nCluster_branch = tree->GetBranch("trk_nCluster");
    if (trk_nCluster_branch) {
      trk_nCluster_branch->SetAddress(&trk_nCluster_);
    }
  }
  trk_bestFromFirstHitSimTrkNChi2_branch = 0;
  if (tree->GetBranch("trk_bestFromFirstHitSimTrkNChi2") != 0) {
    trk_bestFromFirstHitSimTrkNChi2_branch = tree->GetBranch("trk_bestFromFirstHitSimTrkNChi2");
    if (trk_bestFromFirstHitSimTrkNChi2_branch) {
      trk_bestFromFirstHitSimTrkNChi2_branch->SetAddress(&trk_bestFromFirstHitSimTrkNChi2_);
    }
  }
  trk_isHP_branch = 0;
  if (tree->GetBranch("trk_isHP") != 0) {
    trk_isHP_branch = tree->GetBranch("trk_isHP");
    if (trk_isHP_branch) {
      trk_isHP_branch->SetAddress(&trk_isHP_);
    }
  }
  simhit_hitType_branch = 0;
  if (tree->GetBranch("simhit_hitType") != 0) {
    simhit_hitType_branch = tree->GetBranch("simhit_hitType");
    if (simhit_hitType_branch) {
      simhit_hitType_branch->SetAddress(&simhit_hitType_);
    }
  }
  ph2_isUpper_branch = 0;
  if (tree->GetBranch("ph2_isUpper") != 0) {
    ph2_isUpper_branch = tree->GetBranch("ph2_isUpper");
    if (ph2_isUpper_branch) {
      ph2_isUpper_branch->SetAddress(&ph2_isUpper_);
    }
  }
  see_nStrip_branch = 0;
  if (tree->GetBranch("see_nStrip") != 0) {
    see_nStrip_branch = tree->GetBranch("see_nStrip");
    if (see_nStrip_branch) {
      see_nStrip_branch->SetAddress(&see_nStrip_);
    }
  }
  trk_bestSimTrkShareFracSimClusterDenom_branch = 0;
  if (tree->GetBranch("trk_bestSimTrkShareFracSimClusterDenom") != 0) {
    trk_bestSimTrkShareFracSimClusterDenom_branch = tree->GetBranch("trk_bestSimTrkShareFracSimClusterDenom");
    if (trk_bestSimTrkShareFracSimClusterDenom_branch) {
      trk_bestSimTrkShareFracSimClusterDenom_branch->SetAddress(&trk_bestSimTrkShareFracSimClusterDenom_);
    }
  }
  simhit_side_branch = 0;
  if (tree->GetBranch("simhit_side") != 0) {
    simhit_side_branch = tree->GetBranch("simhit_side");
    if (simhit_side_branch) {
      simhit_side_branch->SetAddress(&simhit_side_);
    }
  }
  simhit_x_branch = 0;
  if (tree->GetBranch("simhit_x") != 0) {
    simhit_x_branch = tree->GetBranch("simhit_x");
    if (simhit_x_branch) {
      simhit_x_branch->SetAddress(&simhit_x_);
    }
  }
  see_q_branch = 0;
  if (tree->GetBranch("see_q") != 0) {
    see_q_branch = tree->GetBranch("see_q");
    if (see_q_branch) {
      see_q_branch->SetAddress(&see_q_);
    }
  }
  simhit_z_branch = 0;
  if (tree->GetBranch("simhit_z") != 0) {
    simhit_z_branch = tree->GetBranch("simhit_z");
    if (simhit_z_branch) {
      simhit_z_branch->SetAddress(&simhit_z_);
    }
  }
  sim_pca_lambda_branch = 0;
  if (tree->GetBranch("sim_pca_lambda") != 0) {
    sim_pca_lambda_branch = tree->GetBranch("sim_pca_lambda");
    if (sim_pca_lambda_branch) {
      sim_pca_lambda_branch->SetAddress(&sim_pca_lambda_);
    }
  }
  sim_q_branch = 0;
  if (tree->GetBranch("sim_q") != 0) {
    sim_q_branch = tree->GetBranch("sim_q");
    if (sim_q_branch) {
      sim_q_branch->SetAddress(&sim_q_);
    }
  }
  pix_bbxi_branch = 0;
  if (tree->GetBranch("pix_bbxi") != 0) {
    pix_bbxi_branch = tree->GetBranch("pix_bbxi");
    if (pix_bbxi_branch) {
      pix_bbxi_branch->SetAddress(&pix_bbxi_);
    }
  }
  ph2_order_branch = 0;
  if (tree->GetBranch("ph2_order") != 0) {
    ph2_order_branch = tree->GetBranch("ph2_order");
    if (ph2_order_branch) {
      ph2_order_branch->SetAddress(&ph2_order_);
    }
  }
  ph2_module_branch = 0;
  if (tree->GetBranch("ph2_module") != 0) {
    ph2_module_branch = tree->GetBranch("ph2_module");
    if (ph2_module_branch) {
      ph2_module_branch->SetAddress(&ph2_module_);
    }
  }
  inv_order_branch = 0;
  if (tree->GetBranch("inv_order") != 0) {
    inv_order_branch = tree->GetBranch("inv_order");
    if (inv_order_branch) {
      inv_order_branch->SetAddress(&inv_order_);
    }
  }
  trk_dzErr_branch = 0;
  if (tree->GetBranch("trk_dzErr") != 0) {
    trk_dzErr_branch = tree->GetBranch("trk_dzErr");
    if (trk_dzErr_branch) {
      trk_dzErr_branch->SetAddress(&trk_dzErr_);
    }
  }
  trk_nInnerInactive_branch = 0;
  if (tree->GetBranch("trk_nInnerInactive") != 0) {
    trk_nInnerInactive_branch = tree->GetBranch("trk_nInnerInactive");
    if (trk_nInnerInactive_branch) {
      trk_nInnerInactive_branch->SetAddress(&trk_nInnerInactive_);
    }
  }
  see_fitok_branch = 0;
  if (tree->GetBranch("see_fitok") != 0) {
    see_fitok_branch = tree->GetBranch("see_fitok");
    if (see_fitok_branch) {
      see_fitok_branch->SetAddress(&see_fitok_);
    }
  }
  simhit_blade_branch = 0;
  if (tree->GetBranch("simhit_blade") != 0) {
    simhit_blade_branch = tree->GetBranch("simhit_blade");
    if (simhit_blade_branch) {
      simhit_blade_branch->SetAddress(&simhit_blade_);
    }
  }
  inv_subdet_branch = 0;
  if (tree->GetBranch("inv_subdet") != 0) {
    inv_subdet_branch = tree->GetBranch("inv_subdet");
    if (inv_subdet_branch) {
      inv_subdet_branch->SetAddress(&inv_subdet_);
    }
  }
  pix_blade_branch = 0;
  if (tree->GetBranch("pix_blade") != 0) {
    pix_blade_branch = tree->GetBranch("pix_blade");
    if (pix_blade_branch) {
      pix_blade_branch->SetAddress(&pix_blade_);
    }
  }
  pix_xx_branch = 0;
  if (tree->GetBranch("pix_xx") != 0) {
    pix_xx_branch = tree->GetBranch("pix_xx");
    if (pix_xx_branch) {
      pix_xx_branch->SetAddress(&pix_xx_);
    }
  }
  pix_xy_branch = 0;
  if (tree->GetBranch("pix_xy") != 0) {
    pix_xy_branch = tree->GetBranch("pix_xy");
    if (pix_xy_branch) {
      pix_xy_branch->SetAddress(&pix_xy_);
    }
  }
  simhit_panel_branch = 0;
  if (tree->GetBranch("simhit_panel") != 0) {
    simhit_panel_branch = tree->GetBranch("simhit_panel");
    if (simhit_panel_branch) {
      simhit_panel_branch->SetAddress(&simhit_panel_);
    }
  }
  sim_pz_branch = 0;
  if (tree->GetBranch("sim_pz") != 0) {
    sim_pz_branch = tree->GetBranch("sim_pz");
    if (sim_pz_branch) {
      sim_pz_branch->SetAddress(&sim_pz_);
    }
  }
  trk_dxy_branch = 0;
  if (tree->GetBranch("trk_dxy") != 0) {
    trk_dxy_branch = tree->GetBranch("trk_dxy");
    if (trk_dxy_branch) {
      trk_dxy_branch->SetAddress(&trk_dxy_);
    }
  }
  sim_px_branch = 0;
  if (tree->GetBranch("sim_px") != 0) {
    sim_px_branch = tree->GetBranch("sim_px");
    if (sim_px_branch) {
      sim_px_branch->SetAddress(&sim_px_);
    }
  }
  trk_lambda_branch = 0;
  if (tree->GetBranch("trk_lambda") != 0) {
    trk_lambda_branch = tree->GetBranch("trk_lambda");
    if (trk_lambda_branch) {
      trk_lambda_branch->SetAddress(&trk_lambda_);
    }
  }
  see_stateCcov12_branch = 0;
  if (tree->GetBranch("see_stateCcov12") != 0) {
    see_stateCcov12_branch = tree->GetBranch("see_stateCcov12");
    if (see_stateCcov12_branch) {
      see_stateCcov12_branch->SetAddress(&see_stateCcov12_);
    }
  }
  sim_pt_branch = 0;
  if (tree->GetBranch("sim_pt") != 0) {
    sim_pt_branch = tree->GetBranch("sim_pt");
    if (sim_pt_branch) {
      sim_pt_branch->SetAddress(&sim_pt_);
    }
  }
  sim_py_branch = 0;
  if (tree->GetBranch("sim_py") != 0) {
    sim_py_branch = tree->GetBranch("sim_py");
    if (sim_py_branch) {
      sim_py_branch->SetAddress(&sim_py_);
    }
  }
  sim_decayVtxIdx_branch = 0;
  if (tree->GetBranch("sim_decayVtxIdx") != 0) {
    sim_decayVtxIdx_branch = tree->GetBranch("sim_decayVtxIdx");
    if (sim_decayVtxIdx_branch) {
      sim_decayVtxIdx_branch->SetAddress(&sim_decayVtxIdx_);
    }
  }
  pix_detId_branch = 0;
  if (tree->GetBranch("pix_detId") != 0) {
    pix_detId_branch = tree->GetBranch("pix_detId");
    if (pix_detId_branch) {
      pix_detId_branch->SetAddress(&pix_detId_);
    }
  }
  trk_eta_branch = 0;
  if (tree->GetBranch("trk_eta") != 0) {
    trk_eta_branch = tree->GetBranch("trk_eta");
    if (trk_eta_branch) {
      trk_eta_branch->SetAddress(&trk_eta_);
    }
  }
  see_dxy_branch = 0;
  if (tree->GetBranch("see_dxy") != 0) {
    see_dxy_branch = tree->GetBranch("see_dxy");
    if (see_dxy_branch) {
      see_dxy_branch->SetAddress(&see_dxy_);
    }
  }
  sim_isFromBHadron_branch = 0;
  if (tree->GetBranch("sim_isFromBHadron") != 0) {
    sim_isFromBHadron_branch = tree->GetBranch("sim_isFromBHadron");
    if (sim_isFromBHadron_branch) {
      sim_isFromBHadron_branch->SetAddress(&sim_isFromBHadron_);
    }
  }
  simhit_eloss_branch = 0;
  if (tree->GetBranch("simhit_eloss") != 0) {
    simhit_eloss_branch = tree->GetBranch("simhit_eloss");
    if (simhit_eloss_branch) {
      simhit_eloss_branch->SetAddress(&simhit_eloss_);
    }
  }
  see_stateCcov11_branch = 0;
  if (tree->GetBranch("see_stateCcov11") != 0) {
    see_stateCcov11_branch = tree->GetBranch("see_stateCcov11");
    if (see_stateCcov11_branch) {
      see_stateCcov11_branch->SetAddress(&see_stateCcov11_);
    }
  }
  simhit_pz_branch = 0;
  if (tree->GetBranch("simhit_pz") != 0) {
    simhit_pz_branch = tree->GetBranch("simhit_pz");
    if (simhit_pz_branch) {
      simhit_pz_branch->SetAddress(&simhit_pz_);
    }
  }
  sim_pdgId_branch = 0;
  if (tree->GetBranch("sim_pdgId") != 0) {
    sim_pdgId_branch = tree->GetBranch("sim_pdgId");
    if (sim_pdgId_branch) {
      sim_pdgId_branch->SetAddress(&sim_pdgId_);
    }
  }
  trk_stopReason_branch = 0;
  if (tree->GetBranch("trk_stopReason") != 0) {
    trk_stopReason_branch = tree->GetBranch("trk_stopReason");
    if (trk_stopReason_branch) {
      trk_stopReason_branch->SetAddress(&trk_stopReason_);
    }
  }
  sim_pca_phi_branch = 0;
  if (tree->GetBranch("sim_pca_phi") != 0) {
    sim_pca_phi_branch = tree->GetBranch("sim_pca_phi");
    if (sim_pca_phi_branch) {
      sim_pca_phi_branch->SetAddress(&sim_pca_phi_);
    }
  }
  simhit_isLower_branch = 0;
  if (tree->GetBranch("simhit_isLower") != 0) {
    simhit_isLower_branch = tree->GetBranch("simhit_isLower");
    if (simhit_isLower_branch) {
      simhit_isLower_branch->SetAddress(&simhit_isLower_);
    }
  }
  inv_ring_branch = 0;
  if (tree->GetBranch("inv_ring") != 0) {
    inv_ring_branch = tree->GetBranch("inv_ring");
    if (inv_ring_branch) {
      inv_ring_branch->SetAddress(&inv_ring_);
    }
  }
  ph2_simHitIdx_branch = 0;
  if (tree->GetBranch("ph2_simHitIdx") != 0) {
    ph2_simHitIdx_branch = tree->GetBranch("ph2_simHitIdx");
    if (ph2_simHitIdx_branch) {
      ph2_simHitIdx_branch->SetAddress(&ph2_simHitIdx_);
    }
  }
  simhit_order_branch = 0;
  if (tree->GetBranch("simhit_order") != 0) {
    simhit_order_branch = tree->GetBranch("simhit_order");
    if (simhit_order_branch) {
      simhit_order_branch->SetAddress(&simhit_order_);
    }
  }
  trk_dxyClosestPV_branch = 0;
  if (tree->GetBranch("trk_dxyClosestPV") != 0) {
    trk_dxyClosestPV_branch = tree->GetBranch("trk_dxyClosestPV");
    if (trk_dxyClosestPV_branch) {
      trk_dxyClosestPV_branch->SetAddress(&trk_dxyClosestPV_);
    }
  }
  pix_z_branch = 0;
  if (tree->GetBranch("pix_z") != 0) {
    pix_z_branch = tree->GetBranch("pix_z");
    if (pix_z_branch) {
      pix_z_branch->SetAddress(&pix_z_);
    }
  }
  pix_y_branch = 0;
  if (tree->GetBranch("pix_y") != 0) {
    pix_y_branch = tree->GetBranch("pix_y");
    if (pix_y_branch) {
      pix_y_branch->SetAddress(&pix_y_);
    }
  }
  pix_x_branch = 0;
  if (tree->GetBranch("pix_x") != 0) {
    pix_x_branch = tree->GetBranch("pix_x");
    if (pix_x_branch) {
      pix_x_branch->SetAddress(&pix_x_);
    }
  }
  see_hitType_branch = 0;
  if (tree->GetBranch("see_hitType") != 0) {
    see_hitType_branch = tree->GetBranch("see_hitType");
    if (see_hitType_branch) {
      see_hitType_branch->SetAddress(&see_hitType_);
    }
  }
  see_statePt_branch = 0;
  if (tree->GetBranch("see_statePt") != 0) {
    see_statePt_branch = tree->GetBranch("see_statePt");
    if (see_statePt_branch) {
      see_statePt_branch->SetAddress(&see_statePt_);
    }
  }
  simvtx_sourceSimIdx_branch = 0;
  if (tree->GetBranch("simvtx_sourceSimIdx") != 0) {
    simvtx_sourceSimIdx_branch = tree->GetBranch("simvtx_sourceSimIdx");
    if (simvtx_sourceSimIdx_branch) {
      simvtx_sourceSimIdx_branch->SetAddress(&simvtx_sourceSimIdx_);
    }
  }
  event_branch = 0;
  if (tree->GetBranch("event") != 0) {
    event_branch = tree->GetBranch("event");
    if (event_branch) {
      event_branch->SetAddress(&event_);
    }
  }
  pix_module_branch = 0;
  if (tree->GetBranch("pix_module") != 0) {
    pix_module_branch = tree->GetBranch("pix_module");
    if (pix_module_branch) {
      pix_module_branch->SetAddress(&pix_module_);
    }
  }
  ph2_side_branch = 0;
  if (tree->GetBranch("ph2_side") != 0) {
    ph2_side_branch = tree->GetBranch("ph2_side");
    if (ph2_side_branch) {
      ph2_side_branch->SetAddress(&ph2_side_);
    }
  }
  trk_bestSimTrkNChi2_branch = 0;
  if (tree->GetBranch("trk_bestSimTrkNChi2") != 0) {
    trk_bestSimTrkNChi2_branch = tree->GetBranch("trk_bestSimTrkNChi2");
    if (trk_bestSimTrkNChi2_branch) {
      trk_bestSimTrkNChi2_branch->SetAddress(&trk_bestSimTrkNChi2_);
    }
  }
  see_stateTrajPy_branch = 0;
  if (tree->GetBranch("see_stateTrajPy") != 0) {
    see_stateTrajPy_branch = tree->GetBranch("see_stateTrajPy");
    if (see_stateTrajPy_branch) {
      see_stateTrajPy_branch->SetAddress(&see_stateTrajPy_);
    }
  }
  inv_type_branch = 0;
  if (tree->GetBranch("inv_type") != 0) {
    inv_type_branch = tree->GetBranch("inv_type");
    if (inv_type_branch) {
      inv_type_branch->SetAddress(&inv_type_);
    }
  }
  bsp_z_branch = 0;
  if (tree->GetBranch("bsp_z") != 0) {
    bsp_z_branch = tree->GetBranch("bsp_z");
    if (bsp_z_branch) {
      bsp_z_branch->SetAddress(&bsp_z_);
    }
  }
  bsp_y_branch = 0;
  if (tree->GetBranch("bsp_y") != 0) {
    bsp_y_branch = tree->GetBranch("bsp_y");
    if (bsp_y_branch) {
      bsp_y_branch->SetAddress(&bsp_y_);
    }
  }
  simhit_py_branch = 0;
  if (tree->GetBranch("simhit_py") != 0) {
    simhit_py_branch = tree->GetBranch("simhit_py");
    if (simhit_py_branch) {
      simhit_py_branch->SetAddress(&simhit_py_);
    }
  }
  see_simTrkIdx_branch = 0;
  if (tree->GetBranch("see_simTrkIdx") != 0) {
    see_simTrkIdx_branch = tree->GetBranch("see_simTrkIdx");
    if (see_simTrkIdx_branch) {
      see_simTrkIdx_branch->SetAddress(&see_simTrkIdx_);
    }
  }
  see_stateTrajGlbZ_branch = 0;
  if (tree->GetBranch("see_stateTrajGlbZ") != 0) {
    see_stateTrajGlbZ_branch = tree->GetBranch("see_stateTrajGlbZ");
    if (see_stateTrajGlbZ_branch) {
      see_stateTrajGlbZ_branch->SetAddress(&see_stateTrajGlbZ_);
    }
  }
  see_stateTrajGlbX_branch = 0;
  if (tree->GetBranch("see_stateTrajGlbX") != 0) {
    see_stateTrajGlbX_branch = tree->GetBranch("see_stateTrajGlbX");
    if (see_stateTrajGlbX_branch) {
      see_stateTrajGlbX_branch->SetAddress(&see_stateTrajGlbX_);
    }
  }
  see_stateTrajGlbY_branch = 0;
  if (tree->GetBranch("see_stateTrajGlbY") != 0) {
    see_stateTrajGlbY_branch = tree->GetBranch("see_stateTrajGlbY");
    if (see_stateTrajGlbY_branch) {
      see_stateTrajGlbY_branch->SetAddress(&see_stateTrajGlbY_);
    }
  }
  trk_originalAlgo_branch = 0;
  if (tree->GetBranch("trk_originalAlgo") != 0) {
    trk_originalAlgo_branch = tree->GetBranch("trk_originalAlgo");
    if (trk_originalAlgo_branch) {
      trk_originalAlgo_branch->SetAddress(&trk_originalAlgo_);
    }
  }
  trk_nPixel_branch = 0;
  if (tree->GetBranch("trk_nPixel") != 0) {
    trk_nPixel_branch = tree->GetBranch("trk_nPixel");
    if (trk_nPixel_branch) {
      trk_nPixel_branch->SetAddress(&trk_nPixel_);
    }
  }
  see_stateCcov14_branch = 0;
  if (tree->GetBranch("see_stateCcov14") != 0) {
    see_stateCcov14_branch = tree->GetBranch("see_stateCcov14");
    if (see_stateCcov14_branch) {
      see_stateCcov14_branch->SetAddress(&see_stateCcov14_);
    }
  }
  see_stateCcov15_branch = 0;
  if (tree->GetBranch("see_stateCcov15") != 0) {
    see_stateCcov15_branch = tree->GetBranch("see_stateCcov15");
    if (see_stateCcov15_branch) {
      see_stateCcov15_branch->SetAddress(&see_stateCcov15_);
    }
  }
  trk_phiErr_branch = 0;
  if (tree->GetBranch("trk_phiErr") != 0) {
    trk_phiErr_branch = tree->GetBranch("trk_phiErr");
    if (trk_phiErr_branch) {
      trk_phiErr_branch->SetAddress(&trk_phiErr_);
    }
  }
  see_stateCcov13_branch = 0;
  if (tree->GetBranch("see_stateCcov13") != 0) {
    see_stateCcov13_branch = tree->GetBranch("see_stateCcov13");
    if (see_stateCcov13_branch) {
      see_stateCcov13_branch->SetAddress(&see_stateCcov13_);
    }
  }
  pix_chargeFraction_branch = 0;
  if (tree->GetBranch("pix_chargeFraction") != 0) {
    pix_chargeFraction_branch = tree->GetBranch("pix_chargeFraction");
    if (pix_chargeFraction_branch) {
      pix_chargeFraction_branch->SetAddress(&pix_chargeFraction_);
    }
  }
  trk_q_branch = 0;
  if (tree->GetBranch("trk_q") != 0) {
    trk_q_branch = tree->GetBranch("trk_q");
    if (trk_q_branch) {
      trk_q_branch->SetAddress(&trk_q_);
    }
  }
  sim_seedIdx_branch = 0;
  if (tree->GetBranch("sim_seedIdx") != 0) {
    sim_seedIdx_branch = tree->GetBranch("sim_seedIdx");
    if (sim_seedIdx_branch) {
      sim_seedIdx_branch->SetAddress(&sim_seedIdx_);
    }
  }
  see_dzErr_branch = 0;
  if (tree->GetBranch("see_dzErr") != 0) {
    see_dzErr_branch = tree->GetBranch("see_dzErr");
    if (see_dzErr_branch) {
      see_dzErr_branch->SetAddress(&see_dzErr_);
    }
  }
  sim_nRecoClusters_branch = 0;
  if (tree->GetBranch("sim_nRecoClusters") != 0) {
    sim_nRecoClusters_branch = tree->GetBranch("sim_nRecoClusters");
    if (sim_nRecoClusters_branch) {
      sim_nRecoClusters_branch->SetAddress(&sim_nRecoClusters_);
    }
  }
  run_branch = 0;
  if (tree->GetBranch("run") != 0) {
    run_branch = tree->GetBranch("run");
    if (run_branch) {
      run_branch->SetAddress(&run_);
    }
  }
  ph2_xySignificance_branch = 0;
  if (tree->GetBranch("ph2_xySignificance") != 0) {
    ph2_xySignificance_branch = tree->GetBranch("ph2_xySignificance");
    if (ph2_xySignificance_branch) {
      ph2_xySignificance_branch->SetAddress(&ph2_xySignificance_);
    }
  }
  trk_nChi2_branch = 0;
  if (tree->GetBranch("trk_nChi2") != 0) {
    trk_nChi2_branch = tree->GetBranch("trk_nChi2");
    if (trk_nChi2_branch) {
      trk_nChi2_branch->SetAddress(&trk_nChi2_);
    }
  }
  pix_layer_branch = 0;
  if (tree->GetBranch("pix_layer") != 0) {
    pix_layer_branch = tree->GetBranch("pix_layer");
    if (pix_layer_branch) {
      pix_layer_branch->SetAddress(&pix_layer_);
    }
  }
  pix_xySignificance_branch = 0;
  if (tree->GetBranch("pix_xySignificance") != 0) {
    pix_xySignificance_branch = tree->GetBranch("pix_xySignificance");
    if (pix_xySignificance_branch) {
      pix_xySignificance_branch->SetAddress(&pix_xySignificance_);
    }
  }
  sim_pca_eta_branch = 0;
  if (tree->GetBranch("sim_pca_eta") != 0) {
    sim_pca_eta_branch = tree->GetBranch("sim_pca_eta");
    if (sim_pca_eta_branch) {
      sim_pca_eta_branch->SetAddress(&sim_pca_eta_);
    }
  }
  see_bestSimTrkShareFrac_branch = 0;
  if (tree->GetBranch("see_bestSimTrkShareFrac") != 0) {
    see_bestSimTrkShareFrac_branch = tree->GetBranch("see_bestSimTrkShareFrac");
    if (see_bestSimTrkShareFrac_branch) {
      see_bestSimTrkShareFrac_branch->SetAddress(&see_bestSimTrkShareFrac_);
    }
  }
  see_etaErr_branch = 0;
  if (tree->GetBranch("see_etaErr") != 0) {
    see_etaErr_branch = tree->GetBranch("see_etaErr");
    if (see_etaErr_branch) {
      see_etaErr_branch->SetAddress(&see_etaErr_);
    }
  }
  trk_bestSimTrkShareFracSimDenom_branch = 0;
  if (tree->GetBranch("trk_bestSimTrkShareFracSimDenom") != 0) {
    trk_bestSimTrkShareFracSimDenom_branch = tree->GetBranch("trk_bestSimTrkShareFracSimDenom");
    if (trk_bestSimTrkShareFracSimDenom_branch) {
      trk_bestSimTrkShareFracSimDenom_branch->SetAddress(&trk_bestSimTrkShareFracSimDenom_);
    }
  }
  bsp_sigmaz_branch = 0;
  if (tree->GetBranch("bsp_sigmaz") != 0) {
    bsp_sigmaz_branch = tree->GetBranch("bsp_sigmaz");
    if (bsp_sigmaz_branch) {
      bsp_sigmaz_branch->SetAddress(&bsp_sigmaz_);
    }
  }
  bsp_sigmay_branch = 0;
  if (tree->GetBranch("bsp_sigmay") != 0) {
    bsp_sigmay_branch = tree->GetBranch("bsp_sigmay");
    if (bsp_sigmay_branch) {
      bsp_sigmay_branch->SetAddress(&bsp_sigmay_);
    }
  }
  bsp_sigmax_branch = 0;
  if (tree->GetBranch("bsp_sigmax") != 0) {
    bsp_sigmax_branch = tree->GetBranch("bsp_sigmax");
    if (bsp_sigmax_branch) {
      bsp_sigmax_branch->SetAddress(&bsp_sigmax_);
    }
  }
  pix_ladder_branch = 0;
  if (tree->GetBranch("pix_ladder") != 0) {
    pix_ladder_branch = tree->GetBranch("pix_ladder");
    if (pix_ladder_branch) {
      pix_ladder_branch->SetAddress(&pix_ladder_);
    }
  }
  trk_qualityMask_branch = 0;
  if (tree->GetBranch("trk_qualityMask") != 0) {
    trk_qualityMask_branch = tree->GetBranch("trk_qualityMask");
    if (trk_qualityMask_branch) {
      trk_qualityMask_branch->SetAddress(&trk_qualityMask_);
    }
  }
  trk_ndof_branch = 0;
  if (tree->GetBranch("trk_ndof") != 0) {
    trk_ndof_branch = tree->GetBranch("trk_ndof");
    if (trk_ndof_branch) {
      trk_ndof_branch->SetAddress(&trk_ndof_);
    }
  }
  pix_subdet_branch = 0;
  if (tree->GetBranch("pix_subdet") != 0) {
    pix_subdet_branch = tree->GetBranch("pix_subdet");
    if (pix_subdet_branch) {
      pix_subdet_branch->SetAddress(&pix_subdet_);
    }
  }
  ph2_seeIdx_branch = 0;
  if (tree->GetBranch("ph2_seeIdx") != 0) {
    ph2_seeIdx_branch = tree->GetBranch("ph2_seeIdx");
    if (ph2_seeIdx_branch) {
      ph2_seeIdx_branch->SetAddress(&ph2_seeIdx_);
    }
  }
  inv_isUpper_branch = 0;
  if (tree->GetBranch("inv_isUpper") != 0) {
    inv_isUpper_branch = tree->GetBranch("inv_isUpper");
    if (inv_isUpper_branch) {
      inv_isUpper_branch->SetAddress(&inv_isUpper_);
    }
  }
  ph2_zx_branch = 0;
  if (tree->GetBranch("ph2_zx") != 0) {
    ph2_zx_branch = tree->GetBranch("ph2_zx");
    if (ph2_zx_branch) {
      ph2_zx_branch->SetAddress(&ph2_zx_);
    }
  }
  pix_trkIdx_branch = 0;
  if (tree->GetBranch("pix_trkIdx") != 0) {
    pix_trkIdx_branch = tree->GetBranch("pix_trkIdx");
    if (pix_trkIdx_branch) {
      pix_trkIdx_branch->SetAddress(&pix_trkIdx_);
    }
  }
  trk_nOuterLost_branch = 0;
  if (tree->GetBranch("trk_nOuterLost") != 0) {
    trk_nOuterLost_branch = tree->GetBranch("trk_nOuterLost");
    if (trk_nOuterLost_branch) {
      trk_nOuterLost_branch->SetAddress(&trk_nOuterLost_);
    }
  }
  inv_panel_branch = 0;
  if (tree->GetBranch("inv_panel") != 0) {
    inv_panel_branch = tree->GetBranch("inv_panel");
    if (inv_panel_branch) {
      inv_panel_branch->SetAddress(&inv_panel_);
    }
  }
  vtx_z_branch = 0;
  if (tree->GetBranch("vtx_z") != 0) {
    vtx_z_branch = tree->GetBranch("vtx_z");
    if (vtx_z_branch) {
      vtx_z_branch->SetAddress(&vtx_z_);
    }
  }
  simhit_layer_branch = 0;
  if (tree->GetBranch("simhit_layer") != 0) {
    simhit_layer_branch = tree->GetBranch("simhit_layer");
    if (simhit_layer_branch) {
      simhit_layer_branch->SetAddress(&simhit_layer_);
    }
  }
  vtx_y_branch = 0;
  if (tree->GetBranch("vtx_y") != 0) {
    vtx_y_branch = tree->GetBranch("vtx_y");
    if (vtx_y_branch) {
      vtx_y_branch->SetAddress(&vtx_y_);
    }
  }
  ph2_isBarrel_branch = 0;
  if (tree->GetBranch("ph2_isBarrel") != 0) {
    ph2_isBarrel_branch = tree->GetBranch("ph2_isBarrel");
    if (ph2_isBarrel_branch) {
      ph2_isBarrel_branch->SetAddress(&ph2_isBarrel_);
    }
  }
  pix_seeIdx_branch = 0;
  if (tree->GetBranch("pix_seeIdx") != 0) {
    pix_seeIdx_branch = tree->GetBranch("pix_seeIdx");
    if (pix_seeIdx_branch) {
      pix_seeIdx_branch->SetAddress(&pix_seeIdx_);
    }
  }
  trk_bestFromFirstHitSimTrkIdx_branch = 0;
  if (tree->GetBranch("trk_bestFromFirstHitSimTrkIdx") != 0) {
    trk_bestFromFirstHitSimTrkIdx_branch = tree->GetBranch("trk_bestFromFirstHitSimTrkIdx");
    if (trk_bestFromFirstHitSimTrkIdx_branch) {
      trk_bestFromFirstHitSimTrkIdx_branch->SetAddress(&trk_bestFromFirstHitSimTrkIdx_);
    }
  }
  simhit_px_branch = 0;
  if (tree->GetBranch("simhit_px") != 0) {
    simhit_px_branch = tree->GetBranch("simhit_px");
    if (simhit_px_branch) {
      simhit_px_branch->SetAddress(&simhit_px_);
    }
  }
  see_stateTrajX_branch = 0;
  if (tree->GetBranch("see_stateTrajX") != 0) {
    see_stateTrajX_branch = tree->GetBranch("see_stateTrajX");
    if (see_stateTrajX_branch) {
      see_stateTrajX_branch->SetAddress(&see_stateTrajX_);
    }
  }
  see_stateTrajY_branch = 0;
  if (tree->GetBranch("see_stateTrajY") != 0) {
    see_stateTrajY_branch = tree->GetBranch("see_stateTrajY");
    if (see_stateTrajY_branch) {
      see_stateTrajY_branch->SetAddress(&see_stateTrajY_);
    }
  }
  trk_nOuterInactive_branch = 0;
  if (tree->GetBranch("trk_nOuterInactive") != 0) {
    trk_nOuterInactive_branch = tree->GetBranch("trk_nOuterInactive");
    if (trk_nOuterInactive_branch) {
      trk_nOuterInactive_branch->SetAddress(&trk_nOuterInactive_);
    }
  }
  sim_pca_dxy_branch = 0;
  if (tree->GetBranch("sim_pca_dxy") != 0) {
    sim_pca_dxy_branch = tree->GetBranch("sim_pca_dxy");
    if (sim_pca_dxy_branch) {
      sim_pca_dxy_branch->SetAddress(&sim_pca_dxy_);
    }
  }
  trk_algo_branch = 0;
  if (tree->GetBranch("trk_algo") != 0) {
    trk_algo_branch = tree->GetBranch("trk_algo");
    if (trk_algo_branch) {
      trk_algo_branch->SetAddress(&trk_algo_);
    }
  }
  trk_hitType_branch = 0;
  if (tree->GetBranch("trk_hitType") != 0) {
    trk_hitType_branch = tree->GetBranch("trk_hitType");
    if (trk_hitType_branch) {
      trk_hitType_branch->SetAddress(&trk_hitType_);
    }
  }
  trk_bestFromFirstHitSimTrkShareFrac_branch = 0;
  if (tree->GetBranch("trk_bestFromFirstHitSimTrkShareFrac") != 0) {
    trk_bestFromFirstHitSimTrkShareFrac_branch = tree->GetBranch("trk_bestFromFirstHitSimTrkShareFrac");
    if (trk_bestFromFirstHitSimTrkShareFrac_branch) {
      trk_bestFromFirstHitSimTrkShareFrac_branch->SetAddress(&trk_bestFromFirstHitSimTrkShareFrac_);
    }
  }
  inv_isBarrel_branch = 0;
  if (tree->GetBranch("inv_isBarrel") != 0) {
    inv_isBarrel_branch = tree->GetBranch("inv_isBarrel");
    if (inv_isBarrel_branch) {
      inv_isBarrel_branch->SetAddress(&inv_isBarrel_);
    }
  }
  simvtx_event_branch = 0;
  if (tree->GetBranch("simvtx_event") != 0) {
    simvtx_event_branch = tree->GetBranch("simvtx_event");
    if (simvtx_event_branch) {
      simvtx_event_branch->SetAddress(&simvtx_event_);
    }
  }
  ph2_z_branch = 0;
  if (tree->GetBranch("ph2_z") != 0) {
    ph2_z_branch = tree->GetBranch("ph2_z");
    if (ph2_z_branch) {
      ph2_z_branch->SetAddress(&ph2_z_);
    }
  }
  ph2_x_branch = 0;
  if (tree->GetBranch("ph2_x") != 0) {
    ph2_x_branch = tree->GetBranch("ph2_x");
    if (ph2_x_branch) {
      ph2_x_branch->SetAddress(&ph2_x_);
    }
  }
  ph2_y_branch = 0;
  if (tree->GetBranch("ph2_y") != 0) {
    ph2_y_branch = tree->GetBranch("ph2_y");
    if (ph2_y_branch) {
      ph2_y_branch->SetAddress(&ph2_y_);
    }
  }
  sim_genPdgIds_branch = 0;
  if (tree->GetBranch("sim_genPdgIds") != 0) {
    sim_genPdgIds_branch = tree->GetBranch("sim_genPdgIds");
    if (sim_genPdgIds_branch) {
      sim_genPdgIds_branch->SetAddress(&sim_genPdgIds_);
    }
  }
  trk_mva_branch = 0;
  if (tree->GetBranch("trk_mva") != 0) {
    trk_mva_branch = tree->GetBranch("trk_mva");
    if (trk_mva_branch) {
      trk_mva_branch->SetAddress(&trk_mva_);
    }
  }
  see_stateCcov24_branch = 0;
  if (tree->GetBranch("see_stateCcov24") != 0) {
    see_stateCcov24_branch = tree->GetBranch("see_stateCcov24");
    if (see_stateCcov24_branch) {
      see_stateCcov24_branch->SetAddress(&see_stateCcov24_);
    }
  }
  trk_dzClosestPV_branch = 0;
  if (tree->GetBranch("trk_dzClosestPV") != 0) {
    trk_dzClosestPV_branch = tree->GetBranch("trk_dzClosestPV");
    if (trk_dzClosestPV_branch) {
      trk_dzClosestPV_branch->SetAddress(&trk_dzClosestPV_);
    }
  }
  see_nCluster_branch = 0;
  if (tree->GetBranch("see_nCluster") != 0) {
    see_nCluster_branch = tree->GetBranch("see_nCluster");
    if (see_nCluster_branch) {
      see_nCluster_branch->SetAddress(&see_nCluster_);
    }
  }
  inv_rod_branch = 0;
  if (tree->GetBranch("inv_rod") != 0) {
    inv_rod_branch = tree->GetBranch("inv_rod");
    if (inv_rod_branch) {
      inv_rod_branch->SetAddress(&inv_rod_);
    }
  }
  trk_hitIdx_branch = 0;
  if (tree->GetBranch("trk_hitIdx") != 0) {
    trk_hitIdx_branch = tree->GetBranch("trk_hitIdx");
    if (trk_hitIdx_branch) {
      trk_hitIdx_branch->SetAddress(&trk_hitIdx_);
    }
  }
  see_stateCcov22_branch = 0;
  if (tree->GetBranch("see_stateCcov22") != 0) {
    see_stateCcov22_branch = tree->GetBranch("see_stateCcov22");
    if (see_stateCcov22_branch) {
      see_stateCcov22_branch->SetAddress(&see_stateCcov22_);
    }
  }
  pix_simType_branch = 0;
  if (tree->GetBranch("pix_simType") != 0) {
    pix_simType_branch = tree->GetBranch("pix_simType");
    if (pix_simType_branch) {
      pix_simType_branch->SetAddress(&pix_simType_);
    }
  }
  simhit_ring_branch = 0;
  if (tree->GetBranch("simhit_ring") != 0) {
    simhit_ring_branch = tree->GetBranch("simhit_ring");
    if (simhit_ring_branch) {
      simhit_ring_branch->SetAddress(&simhit_ring_);
    }
  }
  trk_outer_px_branch = 0;
  if (tree->GetBranch("trk_outer_px") != 0) {
    trk_outer_px_branch = tree->GetBranch("trk_outer_px");
    if (trk_outer_px_branch) {
      trk_outer_px_branch->SetAddress(&trk_outer_px_);
    }
  }
  trk_outer_py_branch = 0;
  if (tree->GetBranch("trk_outer_py") != 0) {
    trk_outer_py_branch = tree->GetBranch("trk_outer_py");
    if (trk_outer_py_branch) {
      trk_outer_py_branch->SetAddress(&trk_outer_py_);
    }
  }
  trk_outer_pz_branch = 0;
  if (tree->GetBranch("trk_outer_pz") != 0) {
    trk_outer_pz_branch = tree->GetBranch("trk_outer_pz");
    if (trk_outer_pz_branch) {
      trk_outer_pz_branch->SetAddress(&trk_outer_pz_);
    }
  }
  ph2_zz_branch = 0;
  if (tree->GetBranch("ph2_zz") != 0) {
    ph2_zz_branch = tree->GetBranch("ph2_zz");
    if (ph2_zz_branch) {
      ph2_zz_branch->SetAddress(&ph2_zz_);
    }
  }
  trk_outer_pt_branch = 0;
  if (tree->GetBranch("trk_outer_pt") != 0) {
    trk_outer_pt_branch = tree->GetBranch("trk_outer_pt");
    if (trk_outer_pt_branch) {
      trk_outer_pt_branch->SetAddress(&trk_outer_pt_);
    }
  }
  trk_n3DLay_branch = 0;
  if (tree->GetBranch("trk_n3DLay") != 0) {
    trk_n3DLay_branch = tree->GetBranch("trk_n3DLay");
    if (trk_n3DLay_branch) {
      trk_n3DLay_branch->SetAddress(&trk_n3DLay_);
    }
  }
  trk_nValid_branch = 0;
  if (tree->GetBranch("trk_nValid") != 0) {
    trk_nValid_branch = tree->GetBranch("trk_nValid");
    if (trk_nValid_branch) {
      trk_nValid_branch->SetAddress(&trk_nValid_);
    }
  }
  see_ptErr_branch = 0;
  if (tree->GetBranch("see_ptErr") != 0) {
    see_ptErr_branch = tree->GetBranch("see_ptErr");
    if (see_ptErr_branch) {
      see_ptErr_branch->SetAddress(&see_ptErr_);
    }
  }
  see_stateTrajGlbPx_branch = 0;
  if (tree->GetBranch("see_stateTrajGlbPx") != 0) {
    see_stateTrajGlbPx_branch = tree->GetBranch("see_stateTrajGlbPx");
    if (see_stateTrajGlbPx_branch) {
      see_stateTrajGlbPx_branch->SetAddress(&see_stateTrajGlbPx_);
    }
  }
  ph2_simType_branch = 0;
  if (tree->GetBranch("ph2_simType") != 0) {
    ph2_simType_branch = tree->GetBranch("ph2_simType");
    if (ph2_simType_branch) {
      ph2_simType_branch->SetAddress(&ph2_simType_);
    }
  }
  trk_bestFromFirstHitSimTrkShareFracSimClusterDenom_branch = 0;
  if (tree->GetBranch("trk_bestFromFirstHitSimTrkShareFracSimClusterDenom") != 0) {
    trk_bestFromFirstHitSimTrkShareFracSimClusterDenom_branch =
        tree->GetBranch("trk_bestFromFirstHitSimTrkShareFracSimClusterDenom");
    if (trk_bestFromFirstHitSimTrkShareFracSimClusterDenom_branch) {
      trk_bestFromFirstHitSimTrkShareFracSimClusterDenom_branch->SetAddress(
          &trk_bestFromFirstHitSimTrkShareFracSimClusterDenom_);
    }
  }
  sim_hits_branch = 0;
  if (tree->GetBranch("sim_hits") != 0) {
    sim_hits_branch = tree->GetBranch("sim_hits");
    if (sim_hits_branch) {
      sim_hits_branch->SetAddress(&sim_hits_);
    }
  }
  sim_len_branch = 0;
  if (tree->GetBranch("sim_len") != 0) {
    sim_len_branch = tree->GetBranch("sim_len");
    if (sim_len_branch) {
      sim_len_branch->SetAddress(&sim_len_);
    }
  }
  sim_lengap_branch = 0;
  if (tree->GetBranch("sim_lengap") != 0) {
    sim_lengap_branch = tree->GetBranch("sim_lengap");
    if (sim_lengap_branch) {
      sim_lengap_branch->SetAddress(&sim_lengap_);
    }
  }
  simvtx_x_branch = 0;
  if (tree->GetBranch("simvtx_x") != 0) {
    simvtx_x_branch = tree->GetBranch("simvtx_x");
    if (simvtx_x_branch) {
      simvtx_x_branch->SetAddress(&simvtx_x_);
    }
  }
  trk_pz_branch = 0;
  if (tree->GetBranch("trk_pz") != 0) {
    trk_pz_branch = tree->GetBranch("trk_pz");
    if (trk_pz_branch) {
      trk_pz_branch->SetAddress(&trk_pz_);
    }
  }
  see_bestFromFirstHitSimTrkShareFrac_branch = 0;
  if (tree->GetBranch("see_bestFromFirstHitSimTrkShareFrac") != 0) {
    see_bestFromFirstHitSimTrkShareFrac_branch = tree->GetBranch("see_bestFromFirstHitSimTrkShareFrac");
    if (see_bestFromFirstHitSimTrkShareFrac_branch) {
      see_bestFromFirstHitSimTrkShareFrac_branch->SetAddress(&see_bestFromFirstHitSimTrkShareFrac_);
    }
  }
  trk_px_branch = 0;
  if (tree->GetBranch("trk_px") != 0) {
    trk_px_branch = tree->GetBranch("trk_px");
    if (trk_px_branch) {
      trk_px_branch->SetAddress(&trk_px_);
    }
  }
  trk_py_branch = 0;
  if (tree->GetBranch("trk_py") != 0) {
    trk_py_branch = tree->GetBranch("trk_py");
    if (trk_py_branch) {
      trk_py_branch->SetAddress(&trk_py_);
    }
  }
  trk_vtxIdx_branch = 0;
  if (tree->GetBranch("trk_vtxIdx") != 0) {
    trk_vtxIdx_branch = tree->GetBranch("trk_vtxIdx");
    if (trk_vtxIdx_branch) {
      trk_vtxIdx_branch->SetAddress(&trk_vtxIdx_);
    }
  }
  sim_nPixel_branch = 0;
  if (tree->GetBranch("sim_nPixel") != 0) {
    sim_nPixel_branch = tree->GetBranch("sim_nPixel");
    if (sim_nPixel_branch) {
      sim_nPixel_branch->SetAddress(&sim_nPixel_);
    }
  }
  vtx_chi2_branch = 0;
  if (tree->GetBranch("vtx_chi2") != 0) {
    vtx_chi2_branch = tree->GetBranch("vtx_chi2");
    if (vtx_chi2_branch) {
      vtx_chi2_branch->SetAddress(&vtx_chi2_);
    }
  }
  ph2_ring_branch = 0;
  if (tree->GetBranch("ph2_ring") != 0) {
    ph2_ring_branch = tree->GetBranch("ph2_ring");
    if (ph2_ring_branch) {
      ph2_ring_branch->SetAddress(&ph2_ring_);
    }
  }
  trk_pt_branch = 0;
  if (tree->GetBranch("trk_pt") != 0) {
    trk_pt_branch = tree->GetBranch("trk_pt");
    if (trk_pt_branch) {
      trk_pt_branch->SetAddress(&trk_pt_);
    }
  }
  see_stateCcov44_branch = 0;
  if (tree->GetBranch("see_stateCcov44") != 0) {
    see_stateCcov44_branch = tree->GetBranch("see_stateCcov44");
    if (see_stateCcov44_branch) {
      see_stateCcov44_branch->SetAddress(&see_stateCcov44_);
    }
  }
  ph2_radL_branch = 0;
  if (tree->GetBranch("ph2_radL") != 0) {
    ph2_radL_branch = tree->GetBranch("ph2_radL");
    if (ph2_radL_branch) {
      ph2_radL_branch->SetAddress(&ph2_radL_);
    }
  }
  vtx_zErr_branch = 0;
  if (tree->GetBranch("vtx_zErr") != 0) {
    vtx_zErr_branch = tree->GetBranch("vtx_zErr");
    if (vtx_zErr_branch) {
      vtx_zErr_branch->SetAddress(&vtx_zErr_);
    }
  }
  see_px_branch = 0;
  if (tree->GetBranch("see_px") != 0) {
    see_px_branch = tree->GetBranch("see_px");
    if (see_px_branch) {
      see_px_branch->SetAddress(&see_px_);
    }
  }
  see_pz_branch = 0;
  if (tree->GetBranch("see_pz") != 0) {
    see_pz_branch = tree->GetBranch("see_pz");
    if (see_pz_branch) {
      see_pz_branch->SetAddress(&see_pz_);
    }
  }
  see_eta_branch = 0;
  if (tree->GetBranch("see_eta") != 0) {
    see_eta_branch = tree->GetBranch("see_eta");
    if (see_eta_branch) {
      see_eta_branch->SetAddress(&see_eta_);
    }
  }
  simvtx_bunchCrossing_branch = 0;
  if (tree->GetBranch("simvtx_bunchCrossing") != 0) {
    simvtx_bunchCrossing_branch = tree->GetBranch("simvtx_bunchCrossing");
    if (simvtx_bunchCrossing_branch) {
      simvtx_bunchCrossing_branch->SetAddress(&simvtx_bunchCrossing_);
    }
  }
  sim_pca_dz_branch = 0;
  if (tree->GetBranch("sim_pca_dz") != 0) {
    sim_pca_dz_branch = tree->GetBranch("sim_pca_dz");
    if (sim_pca_dz_branch) {
      sim_pca_dz_branch->SetAddress(&sim_pca_dz_);
    }
  }
  simvtx_y_branch = 0;
  if (tree->GetBranch("simvtx_y") != 0) {
    simvtx_y_branch = tree->GetBranch("simvtx_y");
    if (simvtx_y_branch) {
      simvtx_y_branch->SetAddress(&simvtx_y_);
    }
  }
  inv_isStack_branch = 0;
  if (tree->GetBranch("inv_isStack") != 0) {
    inv_isStack_branch = tree->GetBranch("inv_isStack");
    if (inv_isStack_branch) {
      inv_isStack_branch->SetAddress(&inv_isStack_);
    }
  }
  trk_nStrip_branch = 0;
  if (tree->GetBranch("trk_nStrip") != 0) {
    trk_nStrip_branch = tree->GetBranch("trk_nStrip");
    if (trk_nStrip_branch) {
      trk_nStrip_branch->SetAddress(&trk_nStrip_);
    }
  }
  trk_etaErr_branch = 0;
  if (tree->GetBranch("trk_etaErr") != 0) {
    trk_etaErr_branch = tree->GetBranch("trk_etaErr");
    if (trk_etaErr_branch) {
      trk_etaErr_branch->SetAddress(&trk_etaErr_);
    }
  }
  trk_simTrkNChi2_branch = 0;
  if (tree->GetBranch("trk_simTrkNChi2") != 0) {
    trk_simTrkNChi2_branch = tree->GetBranch("trk_simTrkNChi2");
    if (trk_simTrkNChi2_branch) {
      trk_simTrkNChi2_branch->SetAddress(&trk_simTrkNChi2_);
    }
  }
  pix_zz_branch = 0;
  if (tree->GetBranch("pix_zz") != 0) {
    pix_zz_branch = tree->GetBranch("pix_zz");
    if (pix_zz_branch) {
      pix_zz_branch->SetAddress(&pix_zz_);
    }
  }
  simhit_particle_branch = 0;
  if (tree->GetBranch("simhit_particle") != 0) {
    simhit_particle_branch = tree->GetBranch("simhit_particle");
    if (simhit_particle_branch) {
      simhit_particle_branch->SetAddress(&simhit_particle_);
    }
  }
  see_dz_branch = 0;
  if (tree->GetBranch("see_dz") != 0) {
    see_dz_branch = tree->GetBranch("see_dz");
    if (see_dz_branch) {
      see_dz_branch->SetAddress(&see_dz_);
    }
  }
  see_stateTrajPz_branch = 0;
  if (tree->GetBranch("see_stateTrajPz") != 0) {
    see_stateTrajPz_branch = tree->GetBranch("see_stateTrajPz");
    if (see_stateTrajPz_branch) {
      see_stateTrajPz_branch->SetAddress(&see_stateTrajPz_);
    }
  }
  trk_bestSimTrkShareFrac_branch = 0;
  if (tree->GetBranch("trk_bestSimTrkShareFrac") != 0) {
    trk_bestSimTrkShareFrac_branch = tree->GetBranch("trk_bestSimTrkShareFrac");
    if (trk_bestSimTrkShareFrac_branch) {
      trk_bestSimTrkShareFrac_branch->SetAddress(&trk_bestSimTrkShareFrac_);
    }
  }
  trk_lambdaErr_branch = 0;
  if (tree->GetBranch("trk_lambdaErr") != 0) {
    trk_lambdaErr_branch = tree->GetBranch("trk_lambdaErr");
    if (trk_lambdaErr_branch) {
      trk_lambdaErr_branch->SetAddress(&trk_lambdaErr_);
    }
  }
  see_simTrkShareFrac_branch = 0;
  if (tree->GetBranch("see_simTrkShareFrac") != 0) {
    see_simTrkShareFrac_branch = tree->GetBranch("see_simTrkShareFrac");
    if (see_simTrkShareFrac_branch) {
      see_simTrkShareFrac_branch->SetAddress(&see_simTrkShareFrac_);
    }
  }
  pix_simHitIdx_branch = 0;
  if (tree->GetBranch("pix_simHitIdx") != 0) {
    pix_simHitIdx_branch = tree->GetBranch("pix_simHitIdx");
    if (pix_simHitIdx_branch) {
      pix_simHitIdx_branch->SetAddress(&pix_simHitIdx_);
    }
  }
  vtx_trkIdx_branch = 0;
  if (tree->GetBranch("vtx_trkIdx") != 0) {
    vtx_trkIdx_branch = tree->GetBranch("vtx_trkIdx");
    if (vtx_trkIdx_branch) {
      vtx_trkIdx_branch->SetAddress(&vtx_trkIdx_);
    }
  }
  ph2_rod_branch = 0;
  if (tree->GetBranch("ph2_rod") != 0) {
    ph2_rod_branch = tree->GetBranch("ph2_rod");
    if (ph2_rod_branch) {
      ph2_rod_branch->SetAddress(&ph2_rod_);
    }
  }
  vtx_ndof_branch = 0;
  if (tree->GetBranch("vtx_ndof") != 0) {
    vtx_ndof_branch = tree->GetBranch("vtx_ndof");
    if (vtx_ndof_branch) {
      vtx_ndof_branch->SetAddress(&vtx_ndof_);
    }
  }
  see_nPixel_branch = 0;
  if (tree->GetBranch("see_nPixel") != 0) {
    see_nPixel_branch = tree->GetBranch("see_nPixel");
    if (see_nPixel_branch) {
      see_nPixel_branch->SetAddress(&see_nPixel_);
    }
  }
  sim_nStrip_branch = 0;
  if (tree->GetBranch("sim_nStrip") != 0) {
    sim_nStrip_branch = tree->GetBranch("sim_nStrip");
    if (sim_nStrip_branch) {
      sim_nStrip_branch->SetAddress(&sim_nStrip_);
    }
  }
  sim_bunchCrossing_branch = 0;
  if (tree->GetBranch("sim_bunchCrossing") != 0) {
    sim_bunchCrossing_branch = tree->GetBranch("sim_bunchCrossing");
    if (sim_bunchCrossing_branch) {
      sim_bunchCrossing_branch->SetAddress(&sim_bunchCrossing_);
    }
  }
  see_stateCcov45_branch = 0;
  if (tree->GetBranch("see_stateCcov45") != 0) {
    see_stateCcov45_branch = tree->GetBranch("see_stateCcov45");
    if (see_stateCcov45_branch) {
      see_stateCcov45_branch->SetAddress(&see_stateCcov45_);
    }
  }
  ph2_isStack_branch = 0;
  if (tree->GetBranch("ph2_isStack") != 0) {
    ph2_isStack_branch = tree->GetBranch("ph2_isStack");
    if (ph2_isStack_branch) {
      ph2_isStack_branch->SetAddress(&ph2_isStack_);
    }
  }
  sim_trkShareFrac_branch = 0;
  if (tree->GetBranch("sim_trkShareFrac") != 0) {
    sim_trkShareFrac_branch = tree->GetBranch("sim_trkShareFrac");
    if (sim_trkShareFrac_branch) {
      sim_trkShareFrac_branch->SetAddress(&sim_trkShareFrac_);
    }
  }
  trk_simTrkShareFrac_branch = 0;
  if (tree->GetBranch("trk_simTrkShareFrac") != 0) {
    trk_simTrkShareFrac_branch = tree->GetBranch("trk_simTrkShareFrac");
    if (trk_simTrkShareFrac_branch) {
      trk_simTrkShareFrac_branch->SetAddress(&trk_simTrkShareFrac_);
    }
  }
  sim_phi_branch = 0;
  if (tree->GetBranch("sim_phi") != 0) {
    sim_phi_branch = tree->GetBranch("sim_phi");
    if (sim_phi_branch) {
      sim_phi_branch->SetAddress(&sim_phi_);
    }
  }
  inv_side_branch = 0;
  if (tree->GetBranch("inv_side") != 0) {
    inv_side_branch = tree->GetBranch("inv_side");
    if (inv_side_branch) {
      inv_side_branch->SetAddress(&inv_side_);
    }
  }
  vtx_fake_branch = 0;
  if (tree->GetBranch("vtx_fake") != 0) {
    vtx_fake_branch = tree->GetBranch("vtx_fake");
    if (vtx_fake_branch) {
      vtx_fake_branch->SetAddress(&vtx_fake_);
    }
  }
  trk_nInactive_branch = 0;
  if (tree->GetBranch("trk_nInactive") != 0) {
    trk_nInactive_branch = tree->GetBranch("trk_nInactive");
    if (trk_nInactive_branch) {
      trk_nInactive_branch->SetAddress(&trk_nInactive_);
    }
  }
  trk_nPixelLay_branch = 0;
  if (tree->GetBranch("trk_nPixelLay") != 0) {
    trk_nPixelLay_branch = tree->GetBranch("trk_nPixelLay");
    if (trk_nPixelLay_branch) {
      trk_nPixelLay_branch->SetAddress(&trk_nPixelLay_);
    }
  }
  ph2_bbxi_branch = 0;
  if (tree->GetBranch("ph2_bbxi") != 0) {
    ph2_bbxi_branch = tree->GetBranch("ph2_bbxi");
    if (ph2_bbxi_branch) {
      ph2_bbxi_branch->SetAddress(&ph2_bbxi_);
    }
  }
  vtx_xErr_branch = 0;
  if (tree->GetBranch("vtx_xErr") != 0) {
    vtx_xErr_branch = tree->GetBranch("vtx_xErr");
    if (vtx_xErr_branch) {
      vtx_xErr_branch->SetAddress(&vtx_xErr_);
    }
  }
  see_stateCcov25_branch = 0;
  if (tree->GetBranch("see_stateCcov25") != 0) {
    see_stateCcov25_branch = tree->GetBranch("see_stateCcov25");
    if (see_stateCcov25_branch) {
      see_stateCcov25_branch->SetAddress(&see_stateCcov25_);
    }
  }
  sim_parentVtxIdx_branch = 0;
  if (tree->GetBranch("sim_parentVtxIdx") != 0) {
    sim_parentVtxIdx_branch = tree->GetBranch("sim_parentVtxIdx");
    if (sim_parentVtxIdx_branch) {
      sim_parentVtxIdx_branch->SetAddress(&sim_parentVtxIdx_);
    }
  }
  see_stateCcov23_branch = 0;
  if (tree->GetBranch("see_stateCcov23") != 0) {
    see_stateCcov23_branch = tree->GetBranch("see_stateCcov23");
    if (see_stateCcov23_branch) {
      see_stateCcov23_branch->SetAddress(&see_stateCcov23_);
    }
  }
  trk_algoMask_branch = 0;
  if (tree->GetBranch("trk_algoMask") != 0) {
    trk_algoMask_branch = tree->GetBranch("trk_algoMask");
    if (trk_algoMask_branch) {
      trk_algoMask_branch->SetAddress(&trk_algoMask_);
    }
  }
  trk_simTrkIdx_branch = 0;
  if (tree->GetBranch("trk_simTrkIdx") != 0) {
    trk_simTrkIdx_branch = tree->GetBranch("trk_simTrkIdx");
    if (trk_simTrkIdx_branch) {
      trk_simTrkIdx_branch->SetAddress(&trk_simTrkIdx_);
    }
  }
  see_phiErr_branch = 0;
  if (tree->GetBranch("see_phiErr") != 0) {
    see_phiErr_branch = tree->GetBranch("see_phiErr");
    if (see_phiErr_branch) {
      see_phiErr_branch->SetAddress(&see_phiErr_);
    }
  }
  trk_cotTheta_branch = 0;
  if (tree->GetBranch("trk_cotTheta") != 0) {
    trk_cotTheta_branch = tree->GetBranch("trk_cotTheta");
    if (trk_cotTheta_branch) {
      trk_cotTheta_branch->SetAddress(&trk_cotTheta_);
    }
  }
  see_algo_branch = 0;
  if (tree->GetBranch("see_algo") != 0) {
    see_algo_branch = tree->GetBranch("see_algo");
    if (see_algo_branch) {
      see_algo_branch->SetAddress(&see_algo_);
    }
  }
  simhit_module_branch = 0;
  if (tree->GetBranch("simhit_module") != 0) {
    simhit_module_branch = tree->GetBranch("simhit_module");
    if (simhit_module_branch) {
      simhit_module_branch->SetAddress(&simhit_module_);
    }
  }
  simvtx_daughterSimIdx_branch = 0;
  if (tree->GetBranch("simvtx_daughterSimIdx") != 0) {
    simvtx_daughterSimIdx_branch = tree->GetBranch("simvtx_daughterSimIdx");
    if (simvtx_daughterSimIdx_branch) {
      simvtx_daughterSimIdx_branch->SetAddress(&simvtx_daughterSimIdx_);
    }
  }
  vtx_x_branch = 0;
  if (tree->GetBranch("vtx_x") != 0) {
    vtx_x_branch = tree->GetBranch("vtx_x");
    if (vtx_x_branch) {
      vtx_x_branch->SetAddress(&vtx_x_);
    }
  }
  trk_seedIdx_branch = 0;
  if (tree->GetBranch("trk_seedIdx") != 0) {
    trk_seedIdx_branch = tree->GetBranch("trk_seedIdx");
    if (trk_seedIdx_branch) {
      trk_seedIdx_branch->SetAddress(&trk_seedIdx_);
    }
  }
  simhit_y_branch = 0;
  if (tree->GetBranch("simhit_y") != 0) {
    simhit_y_branch = tree->GetBranch("simhit_y");
    if (simhit_y_branch) {
      simhit_y_branch->SetAddress(&simhit_y_);
    }
  }
  inv_layer_branch = 0;
  if (tree->GetBranch("inv_layer") != 0) {
    inv_layer_branch = tree->GetBranch("inv_layer");
    if (inv_layer_branch) {
      inv_layer_branch->SetAddress(&inv_layer_);
    }
  }
  trk_nLostLay_branch = 0;
  if (tree->GetBranch("trk_nLostLay") != 0) {
    trk_nLostLay_branch = tree->GetBranch("trk_nLostLay");
    if (trk_nLostLay_branch) {
      trk_nLostLay_branch->SetAddress(&trk_nLostLay_);
    }
  }
  ph2_isLower_branch = 0;
  if (tree->GetBranch("ph2_isLower") != 0) {
    ph2_isLower_branch = tree->GetBranch("ph2_isLower");
    if (ph2_isLower_branch) {
      ph2_isLower_branch->SetAddress(&ph2_isLower_);
    }
  }
  pix_side_branch = 0;
  if (tree->GetBranch("pix_side") != 0) {
    pix_side_branch = tree->GetBranch("pix_side");
    if (pix_side_branch) {
      pix_side_branch->SetAddress(&pix_side_);
    }
  }
  inv_isLower_branch = 0;
  if (tree->GetBranch("inv_isLower") != 0) {
    inv_isLower_branch = tree->GetBranch("inv_isLower");
    if (inv_isLower_branch) {
      inv_isLower_branch->SetAddress(&inv_isLower_);
    }
  }
  ph2_trkIdx_branch = 0;
  if (tree->GetBranch("ph2_trkIdx") != 0) {
    ph2_trkIdx_branch = tree->GetBranch("ph2_trkIdx");
    if (ph2_trkIdx_branch) {
      ph2_trkIdx_branch->SetAddress(&ph2_trkIdx_);
    }
  }
  sim_nValid_branch = 0;
  if (tree->GetBranch("sim_nValid") != 0) {
    sim_nValid_branch = tree->GetBranch("sim_nValid");
    if (sim_nValid_branch) {
      sim_nValid_branch->SetAddress(&sim_nValid_);
    }
  }
  simhit_simTrkIdx_branch = 0;
  if (tree->GetBranch("simhit_simTrkIdx") != 0) {
    simhit_simTrkIdx_branch = tree->GetBranch("simhit_simTrkIdx");
    if (simhit_simTrkIdx_branch) {
      simhit_simTrkIdx_branch->SetAddress(&simhit_simTrkIdx_);
    }
  }
  see_nCands_branch = 0;
  if (tree->GetBranch("see_nCands") != 0) {
    see_nCands_branch = tree->GetBranch("see_nCands");
    if (see_nCands_branch) {
      see_nCands_branch->SetAddress(&see_nCands_);
    }
  }
  see_bestSimTrkIdx_branch = 0;
  if (tree->GetBranch("see_bestSimTrkIdx") != 0) {
    see_bestSimTrkIdx_branch = tree->GetBranch("see_bestSimTrkIdx");
    if (see_bestSimTrkIdx_branch) {
      see_bestSimTrkIdx_branch->SetAddress(&see_bestSimTrkIdx_);
    }
  }
  vtx_yErr_branch = 0;
  if (tree->GetBranch("vtx_yErr") != 0) {
    vtx_yErr_branch = tree->GetBranch("vtx_yErr");
    if (vtx_yErr_branch) {
      vtx_yErr_branch->SetAddress(&vtx_yErr_);
    }
  }
  trk_dzPV_branch = 0;
  if (tree->GetBranch("trk_dzPV") != 0) {
    trk_dzPV_branch = tree->GetBranch("trk_dzPV");
    if (trk_dzPV_branch) {
      trk_dzPV_branch->SetAddress(&trk_dzPV_);
    }
  }
  ph2_xy_branch = 0;
  if (tree->GetBranch("ph2_xy") != 0) {
    ph2_xy_branch = tree->GetBranch("ph2_xy");
    if (ph2_xy_branch) {
      ph2_xy_branch->SetAddress(&ph2_xy_);
    }
  }
  inv_module_branch = 0;
  if (tree->GetBranch("inv_module") != 0) {
    inv_module_branch = tree->GetBranch("inv_module");
    if (inv_module_branch) {
      inv_module_branch->SetAddress(&inv_module_);
    }
  }
  see_stateCcov55_branch = 0;
  if (tree->GetBranch("see_stateCcov55") != 0) {
    see_stateCcov55_branch = tree->GetBranch("see_stateCcov55");
    if (see_stateCcov55_branch) {
      see_stateCcov55_branch->SetAddress(&see_stateCcov55_);
    }
  }
  pix_panel_branch = 0;
  if (tree->GetBranch("pix_panel") != 0) {
    pix_panel_branch = tree->GetBranch("pix_panel");
    if (pix_panel_branch) {
      pix_panel_branch->SetAddress(&pix_panel_);
    }
  }
  inv_ladder_branch = 0;
  if (tree->GetBranch("inv_ladder") != 0) {
    inv_ladder_branch = tree->GetBranch("inv_ladder");
    if (inv_ladder_branch) {
      inv_ladder_branch->SetAddress(&inv_ladder_);
    }
  }
  ph2_xx_branch = 0;
  if (tree->GetBranch("ph2_xx") != 0) {
    ph2_xx_branch = tree->GetBranch("ph2_xx");
    if (ph2_xx_branch) {
      ph2_xx_branch->SetAddress(&ph2_xx_);
    }
  }
  sim_pca_cotTheta_branch = 0;
  if (tree->GetBranch("sim_pca_cotTheta") != 0) {
    sim_pca_cotTheta_branch = tree->GetBranch("sim_pca_cotTheta");
    if (sim_pca_cotTheta_branch) {
      sim_pca_cotTheta_branch->SetAddress(&sim_pca_cotTheta_);
    }
  }
  simpv_idx_branch = 0;
  if (tree->GetBranch("simpv_idx") != 0) {
    simpv_idx_branch = tree->GetBranch("simpv_idx");
    if (simpv_idx_branch) {
      simpv_idx_branch->SetAddress(&simpv_idx_);
    }
  }
  trk_inner_pz_branch = 0;
  if (tree->GetBranch("trk_inner_pz") != 0) {
    trk_inner_pz_branch = tree->GetBranch("trk_inner_pz");
    if (trk_inner_pz_branch) {
      trk_inner_pz_branch->SetAddress(&trk_inner_pz_);
    }
  }
  see_chi2_branch = 0;
  if (tree->GetBranch("see_chi2") != 0) {
    see_chi2_branch = tree->GetBranch("see_chi2");
    if (see_chi2_branch) {
      see_chi2_branch->SetAddress(&see_chi2_);
    }
  }
  see_stateCcov35_branch = 0;
  if (tree->GetBranch("see_stateCcov35") != 0) {
    see_stateCcov35_branch = tree->GetBranch("see_stateCcov35");
    if (see_stateCcov35_branch) {
      see_stateCcov35_branch->SetAddress(&see_stateCcov35_);
    }
  }
  see_stateCcov33_branch = 0;
  if (tree->GetBranch("see_stateCcov33") != 0) {
    see_stateCcov33_branch = tree->GetBranch("see_stateCcov33");
    if (see_stateCcov33_branch) {
      see_stateCcov33_branch->SetAddress(&see_stateCcov33_);
    }
  }
  inv_detId_branch = 0;
  if (tree->GetBranch("inv_detId") != 0) {
    inv_detId_branch = tree->GetBranch("inv_detId");
    if (inv_detId_branch) {
      inv_detId_branch->SetAddress(&inv_detId_);
    }
  }
  see_offset_branch = 0;
  if (tree->GetBranch("see_offset") != 0) {
    see_offset_branch = tree->GetBranch("see_offset");
    if (see_offset_branch) {
      see_offset_branch->SetAddress(&see_offset_);
    }
  }
  sim_nLay_branch = 0;
  if (tree->GetBranch("sim_nLay") != 0) {
    sim_nLay_branch = tree->GetBranch("sim_nLay");
    if (sim_nLay_branch) {
      sim_nLay_branch->SetAddress(&sim_nLay_);
    }
  }
  sim_simHitIdx_branch = 0;
  if (tree->GetBranch("sim_simHitIdx") != 0) {
    sim_simHitIdx_branch = tree->GetBranch("sim_simHitIdx");
    if (sim_simHitIdx_branch) {
      sim_simHitIdx_branch->SetAddress(&sim_simHitIdx_);
    }
  }
  simhit_isUpper_branch = 0;
  if (tree->GetBranch("simhit_isUpper") != 0) {
    simhit_isUpper_branch = tree->GetBranch("simhit_isUpper");
    if (simhit_isUpper_branch) {
      simhit_isUpper_branch->SetAddress(&simhit_isUpper_);
    }
  }
  see_stateCcov00_branch = 0;
  if (tree->GetBranch("see_stateCcov00") != 0) {
    see_stateCcov00_branch = tree->GetBranch("see_stateCcov00");
    if (see_stateCcov00_branch) {
      see_stateCcov00_branch->SetAddress(&see_stateCcov00_);
    }
  }
  see_stopReason_branch = 0;
  if (tree->GetBranch("see_stopReason") != 0) {
    see_stopReason_branch = tree->GetBranch("see_stopReason");
    if (see_stopReason_branch) {
      see_stopReason_branch->SetAddress(&see_stopReason_);
    }
  }
  vtx_valid_branch = 0;
  if (tree->GetBranch("vtx_valid") != 0) {
    vtx_valid_branch = tree->GetBranch("vtx_valid");
    if (vtx_valid_branch) {
      vtx_valid_branch->SetAddress(&vtx_valid_);
    }
  }
  lumi_branch = 0;
  if (tree->GetBranch("lumi") != 0) {
    lumi_branch = tree->GetBranch("lumi");
    if (lumi_branch) {
      lumi_branch->SetAddress(&lumi_);
    }
  }
  trk_refpoint_x_branch = 0;
  if (tree->GetBranch("trk_refpoint_x") != 0) {
    trk_refpoint_x_branch = tree->GetBranch("trk_refpoint_x");
    if (trk_refpoint_x_branch) {
      trk_refpoint_x_branch->SetAddress(&trk_refpoint_x_);
    }
  }
  trk_refpoint_y_branch = 0;
  if (tree->GetBranch("trk_refpoint_y") != 0) {
    trk_refpoint_y_branch = tree->GetBranch("trk_refpoint_y");
    if (trk_refpoint_y_branch) {
      trk_refpoint_y_branch->SetAddress(&trk_refpoint_y_);
    }
  }
  trk_refpoint_z_branch = 0;
  if (tree->GetBranch("trk_refpoint_z") != 0) {
    trk_refpoint_z_branch = tree->GetBranch("trk_refpoint_z");
    if (trk_refpoint_z_branch) {
      trk_refpoint_z_branch->SetAddress(&trk_refpoint_z_);
    }
  }
  sim_n3DLay_branch = 0;
  if (tree->GetBranch("sim_n3DLay") != 0) {
    sim_n3DLay_branch = tree->GetBranch("sim_n3DLay");
    if (sim_n3DLay_branch) {
      sim_n3DLay_branch->SetAddress(&sim_n3DLay_);
    }
  }
  see_nPhase2OT_branch = 0;
  if (tree->GetBranch("see_nPhase2OT") != 0) {
    see_nPhase2OT_branch = tree->GetBranch("see_nPhase2OT");
    if (see_nPhase2OT_branch) {
      see_nPhase2OT_branch->SetAddress(&see_nPhase2OT_);
    }
  }
  trk_bestFromFirstHitSimTrkShareFracSimDenom_branch = 0;
  if (tree->GetBranch("trk_bestFromFirstHitSimTrkShareFracSimDenom") != 0) {
    trk_bestFromFirstHitSimTrkShareFracSimDenom_branch = tree->GetBranch("trk_bestFromFirstHitSimTrkShareFracSimDenom");
    if (trk_bestFromFirstHitSimTrkShareFracSimDenom_branch) {
      trk_bestFromFirstHitSimTrkShareFracSimDenom_branch->SetAddress(&trk_bestFromFirstHitSimTrkShareFracSimDenom_);
    }
  }
  ph2_yy_branch = 0;
  if (tree->GetBranch("ph2_yy") != 0) {
    ph2_yy_branch = tree->GetBranch("ph2_yy");
    if (ph2_yy_branch) {
      ph2_yy_branch->SetAddress(&ph2_yy_);
    }
  }
  ph2_yz_branch = 0;
  if (tree->GetBranch("ph2_yz") != 0) {
    ph2_yz_branch = tree->GetBranch("ph2_yz");
    if (ph2_yz_branch) {
      ph2_yz_branch->SetAddress(&ph2_yz_);
    }
  }
  inv_blade_branch = 0;
  if (tree->GetBranch("inv_blade") != 0) {
    inv_blade_branch = tree->GetBranch("inv_blade");
    if (inv_blade_branch) {
      inv_blade_branch->SetAddress(&inv_blade_);
    }
  }
  trk_ptErr_branch = 0;
  if (tree->GetBranch("trk_ptErr") != 0) {
    trk_ptErr_branch = tree->GetBranch("trk_ptErr");
    if (trk_ptErr_branch) {
      trk_ptErr_branch->SetAddress(&trk_ptErr_);
    }
  }
  pix_zx_branch = 0;
  if (tree->GetBranch("pix_zx") != 0) {
    pix_zx_branch = tree->GetBranch("pix_zx");
    if (pix_zx_branch) {
      pix_zx_branch->SetAddress(&pix_zx_);
    }
  }
  simvtx_z_branch = 0;
  if (tree->GetBranch("simvtx_z") != 0) {
    simvtx_z_branch = tree->GetBranch("simvtx_z");
    if (simvtx_z_branch) {
      simvtx_z_branch->SetAddress(&simvtx_z_);
    }
  }
  sim_nTrackerHits_branch = 0;
  if (tree->GetBranch("sim_nTrackerHits") != 0) {
    sim_nTrackerHits_branch = tree->GetBranch("sim_nTrackerHits");
    if (sim_nTrackerHits_branch) {
      sim_nTrackerHits_branch->SetAddress(&sim_nTrackerHits_);
    }
  }
  ph2_subdet_branch = 0;
  if (tree->GetBranch("ph2_subdet") != 0) {
    ph2_subdet_branch = tree->GetBranch("ph2_subdet");
    if (ph2_subdet_branch) {
      ph2_subdet_branch->SetAddress(&ph2_subdet_);
    }
  }
  see_stateTrajPx_branch = 0;
  if (tree->GetBranch("see_stateTrajPx") != 0) {
    see_stateTrajPx_branch = tree->GetBranch("see_stateTrajPx");
    if (see_stateTrajPx_branch) {
      see_stateTrajPx_branch->SetAddress(&see_stateTrajPx_);
    }
  }
  simhit_hitIdx_branch = 0;
  if (tree->GetBranch("simhit_hitIdx") != 0) {
    simhit_hitIdx_branch = tree->GetBranch("simhit_hitIdx");
    if (simhit_hitIdx_branch) {
      simhit_hitIdx_branch->SetAddress(&simhit_hitIdx_);
    }
  }
  simhit_ladder_branch = 0;
  if (tree->GetBranch("simhit_ladder") != 0) {
    simhit_ladder_branch = tree->GetBranch("simhit_ladder");
    if (simhit_ladder_branch) {
      simhit_ladder_branch->SetAddress(&simhit_ladder_);
    }
  }
  ph2_layer_branch = 0;
  if (tree->GetBranch("ph2_layer") != 0) {
    ph2_layer_branch = tree->GetBranch("ph2_layer");
    if (ph2_layer_branch) {
      ph2_layer_branch->SetAddress(&ph2_layer_);
    }
  }
  see_phi_branch = 0;
  if (tree->GetBranch("see_phi") != 0) {
    see_phi_branch = tree->GetBranch("see_phi");
    if (see_phi_branch) {
      see_phi_branch->SetAddress(&see_phi_);
    }
  }
  trk_nChi2_1Dmod_branch = 0;
  if (tree->GetBranch("trk_nChi2_1Dmod") != 0) {
    trk_nChi2_1Dmod_branch = tree->GetBranch("trk_nChi2_1Dmod");
    if (trk_nChi2_1Dmod_branch) {
      trk_nChi2_1Dmod_branch->SetAddress(&trk_nChi2_1Dmod_);
    }
  }
  trk_inner_py_branch = 0;
  if (tree->GetBranch("trk_inner_py") != 0) {
    trk_inner_py_branch = tree->GetBranch("trk_inner_py");
    if (trk_inner_py_branch) {
      trk_inner_py_branch->SetAddress(&trk_inner_py_);
    }
  }
  trk_inner_px_branch = 0;
  if (tree->GetBranch("trk_inner_px") != 0) {
    trk_inner_px_branch = tree->GetBranch("trk_inner_px");
    if (trk_inner_px_branch) {
      trk_inner_px_branch->SetAddress(&trk_inner_px_);
    }
  }
  trk_dxyErr_branch = 0;
  if (tree->GetBranch("trk_dxyErr") != 0) {
    trk_dxyErr_branch = tree->GetBranch("trk_dxyErr");
    if (trk_dxyErr_branch) {
      trk_dxyErr_branch->SetAddress(&trk_dxyErr_);
    }
  }
  sim_nPixelLay_branch = 0;
  if (tree->GetBranch("sim_nPixelLay") != 0) {
    sim_nPixelLay_branch = tree->GetBranch("sim_nPixelLay");
    if (sim_nPixelLay_branch) {
      sim_nPixelLay_branch->SetAddress(&sim_nPixelLay_);
    }
  }
  see_nValid_branch = 0;
  if (tree->GetBranch("see_nValid") != 0) {
    see_nValid_branch = tree->GetBranch("see_nValid");
    if (see_nValid_branch) {
      see_nValid_branch->SetAddress(&see_nValid_);
    }
  }
  trk_inner_pt_branch = 0;
  if (tree->GetBranch("trk_inner_pt") != 0) {
    trk_inner_pt_branch = tree->GetBranch("trk_inner_pt");
    if (trk_inner_pt_branch) {
      trk_inner_pt_branch->SetAddress(&trk_inner_pt_);
    }
  }
  see_stateTrajGlbPy_branch = 0;
  if (tree->GetBranch("see_stateTrajGlbPy") != 0) {
    see_stateTrajGlbPy_branch = tree->GetBranch("see_stateTrajGlbPy");
    if (see_stateTrajGlbPy_branch) {
      see_stateTrajGlbPy_branch->SetAddress(&see_stateTrajGlbPy_);
    }
  }
  tree->SetMakeClass(0);
}
void Trktree::GetEntry(unsigned int idx) {

  sim_etadiffs_isLoaded = false; // Added by Kasia
  sim_phidiffs_isLoaded = false; // Added by Kasia
  sim_rjet_isLoaded = false; // Added by Kasia
  sim_jet_eta_isLoaded = false; // Added by Kasia
  sim_jet_phi_isLoaded = false; // Added by Kasia
  sim_jet_pt_isLoaded = false; // Added by Kasia

  index = idx;
  see_stateCcov01_isLoaded = false;
  simhit_rod_isLoaded = false;
  trk_phi_isLoaded = false;
  bsp_x_isLoaded = false;
  see_stateCcov05_isLoaded = false;
  see_stateCcov04_isLoaded = false;
  trk_dxyPV_isLoaded = false;
  simhit_tof_isLoaded = false;
  sim_event_isLoaded = false;
  simhit_isStack_isLoaded = false;
  trk_dz_isLoaded = false;
  see_stateCcov03_isLoaded = false;
  sim_eta_isLoaded = false;
  simvtx_processType_isLoaded = false;
  pix_radL_isLoaded = false;
  see_stateCcov02_isLoaded = false;
  see_nGlued_isLoaded = false;
  trk_bestSimTrkIdx_isLoaded = false;
  see_stateTrajGlbPz_isLoaded = false;
  pix_yz_isLoaded = false;
  pix_yy_isLoaded = false;
  simhit_process_isLoaded = false;
  see_stateCcov34_isLoaded = false;
  trk_nInnerLost_isLoaded = false;
  see_py_isLoaded = false;
  sim_trkIdx_isLoaded = false;
  trk_nLost_isLoaded = false;
  pix_isBarrel_isLoaded = false;
  see_dxyErr_isLoaded = false;
  simhit_detId_isLoaded = false;
  simhit_subdet_isLoaded = false;
  see_hitIdx_isLoaded = false;
  see_pt_isLoaded = false;
  ph2_detId_isLoaded = false;
  trk_nStripLay_isLoaded = false;
  see_bestFromFirstHitSimTrkIdx_isLoaded = false;
  sim_pca_pt_isLoaded = false;
  see_trkIdx_isLoaded = false;
  trk_nCluster_isLoaded = false;
  trk_bestFromFirstHitSimTrkNChi2_isLoaded = false;
  trk_isHP_isLoaded = false;
  simhit_hitType_isLoaded = false;
  ph2_isUpper_isLoaded = false;
  see_nStrip_isLoaded = false;
  trk_bestSimTrkShareFracSimClusterDenom_isLoaded = false;
  simhit_side_isLoaded = false;
  simhit_x_isLoaded = false;
  see_q_isLoaded = false;
  simhit_z_isLoaded = false;
  sim_pca_lambda_isLoaded = false;
  sim_q_isLoaded = false;
  pix_bbxi_isLoaded = false;
  ph2_order_isLoaded = false;
  ph2_module_isLoaded = false;
  inv_order_isLoaded = false;
  trk_dzErr_isLoaded = false;
  trk_nInnerInactive_isLoaded = false;
  see_fitok_isLoaded = false;
  simhit_blade_isLoaded = false;
  inv_subdet_isLoaded = false;
  pix_blade_isLoaded = false;
  pix_xx_isLoaded = false;
  pix_xy_isLoaded = false;
  simhit_panel_isLoaded = false;
  sim_pz_isLoaded = false;
  trk_dxy_isLoaded = false;
  sim_px_isLoaded = false;
  trk_lambda_isLoaded = false;
  see_stateCcov12_isLoaded = false;
  sim_pt_isLoaded = false;
  sim_py_isLoaded = false;
  sim_decayVtxIdx_isLoaded = false;
  pix_detId_isLoaded = false;
  trk_eta_isLoaded = false;
  see_dxy_isLoaded = false;
  sim_isFromBHadron_isLoaded = false;
  simhit_eloss_isLoaded = false;
  see_stateCcov11_isLoaded = false;
  simhit_pz_isLoaded = false;
  sim_pdgId_isLoaded = false;
  trk_stopReason_isLoaded = false;
  sim_pca_phi_isLoaded = false;
  simhit_isLower_isLoaded = false;
  inv_ring_isLoaded = false;
  ph2_simHitIdx_isLoaded = false;
  simhit_order_isLoaded = false;
  trk_dxyClosestPV_isLoaded = false;
  pix_z_isLoaded = false;
  pix_y_isLoaded = false;
  pix_x_isLoaded = false;
  see_hitType_isLoaded = false;
  see_statePt_isLoaded = false;
  simvtx_sourceSimIdx_isLoaded = false;
  event_isLoaded = false;
  pix_module_isLoaded = false;
  ph2_side_isLoaded = false;
  trk_bestSimTrkNChi2_isLoaded = false;
  see_stateTrajPy_isLoaded = false;
  inv_type_isLoaded = false;
  bsp_z_isLoaded = false;
  bsp_y_isLoaded = false;
  simhit_py_isLoaded = false;
  see_simTrkIdx_isLoaded = false;
  see_stateTrajGlbZ_isLoaded = false;
  see_stateTrajGlbX_isLoaded = false;
  see_stateTrajGlbY_isLoaded = false;
  trk_originalAlgo_isLoaded = false;
  trk_nPixel_isLoaded = false;
  see_stateCcov14_isLoaded = false;
  see_stateCcov15_isLoaded = false;
  trk_phiErr_isLoaded = false;
  see_stateCcov13_isLoaded = false;
  pix_chargeFraction_isLoaded = false;
  trk_q_isLoaded = false;
  sim_seedIdx_isLoaded = false;
  see_dzErr_isLoaded = false;
  sim_nRecoClusters_isLoaded = false;
  run_isLoaded = false;
  ph2_xySignificance_isLoaded = false;
  trk_nChi2_isLoaded = false;
  pix_layer_isLoaded = false;
  pix_xySignificance_isLoaded = false;
  sim_pca_eta_isLoaded = false;
  see_bestSimTrkShareFrac_isLoaded = false;
  see_etaErr_isLoaded = false;
  trk_bestSimTrkShareFracSimDenom_isLoaded = false;
  bsp_sigmaz_isLoaded = false;
  bsp_sigmay_isLoaded = false;
  bsp_sigmax_isLoaded = false;
  pix_ladder_isLoaded = false;
  trk_qualityMask_isLoaded = false;
  trk_ndof_isLoaded = false;
  pix_subdet_isLoaded = false;
  ph2_seeIdx_isLoaded = false;
  inv_isUpper_isLoaded = false;
  ph2_zx_isLoaded = false;
  pix_trkIdx_isLoaded = false;
  trk_nOuterLost_isLoaded = false;
  inv_panel_isLoaded = false;
  vtx_z_isLoaded = false;
  simhit_layer_isLoaded = false;
  vtx_y_isLoaded = false;
  ph2_isBarrel_isLoaded = false;
  pix_seeIdx_isLoaded = false;
  trk_bestFromFirstHitSimTrkIdx_isLoaded = false;
  simhit_px_isLoaded = false;
  see_stateTrajX_isLoaded = false;
  see_stateTrajY_isLoaded = false;
  trk_nOuterInactive_isLoaded = false;
  sim_pca_dxy_isLoaded = false;
  trk_algo_isLoaded = false;
  trk_hitType_isLoaded = false;
  trk_bestFromFirstHitSimTrkShareFrac_isLoaded = false;
  inv_isBarrel_isLoaded = false;
  simvtx_event_isLoaded = false;
  ph2_z_isLoaded = false;
  ph2_x_isLoaded = false;
  ph2_y_isLoaded = false;
  sim_genPdgIds_isLoaded = false;
  trk_mva_isLoaded = false;
  see_stateCcov24_isLoaded = false;
  trk_dzClosestPV_isLoaded = false;
  see_nCluster_isLoaded = false;
  inv_rod_isLoaded = false;
  trk_hitIdx_isLoaded = false;
  see_stateCcov22_isLoaded = false;
  pix_simType_isLoaded = false;
  simhit_ring_isLoaded = false;
  trk_outer_px_isLoaded = false;
  trk_outer_py_isLoaded = false;
  trk_outer_pz_isLoaded = false;
  ph2_zz_isLoaded = false;
  trk_outer_pt_isLoaded = false;
  trk_n3DLay_isLoaded = false;
  trk_nValid_isLoaded = false;
  see_ptErr_isLoaded = false;
  see_stateTrajGlbPx_isLoaded = false;
  ph2_simType_isLoaded = false;
  trk_bestFromFirstHitSimTrkShareFracSimClusterDenom_isLoaded = false;
  sim_hits_isLoaded = false;
  sim_len_isLoaded = false;
  sim_lengap_isLoaded = false;
  simvtx_x_isLoaded = false;
  trk_pz_isLoaded = false;
  see_bestFromFirstHitSimTrkShareFrac_isLoaded = false;
  trk_px_isLoaded = false;
  trk_py_isLoaded = false;
  trk_vtxIdx_isLoaded = false;
  sim_nPixel_isLoaded = false;
  vtx_chi2_isLoaded = false;
  ph2_ring_isLoaded = false;
  trk_pt_isLoaded = false;
  see_stateCcov44_isLoaded = false;
  ph2_radL_isLoaded = false;
  vtx_zErr_isLoaded = false;
  see_px_isLoaded = false;
  see_pz_isLoaded = false;
  see_eta_isLoaded = false;
  simvtx_bunchCrossing_isLoaded = false;
  sim_pca_dz_isLoaded = false;
  simvtx_y_isLoaded = false;
  inv_isStack_isLoaded = false;
  trk_nStrip_isLoaded = false;
  trk_etaErr_isLoaded = false;
  trk_simTrkNChi2_isLoaded = false;
  pix_zz_isLoaded = false;
  simhit_particle_isLoaded = false;
  see_dz_isLoaded = false;
  see_stateTrajPz_isLoaded = false;
  trk_bestSimTrkShareFrac_isLoaded = false;
  trk_lambdaErr_isLoaded = false;
  see_simTrkShareFrac_isLoaded = false;
  pix_simHitIdx_isLoaded = false;
  vtx_trkIdx_isLoaded = false;
  ph2_rod_isLoaded = false;
  vtx_ndof_isLoaded = false;
  see_nPixel_isLoaded = false;
  sim_nStrip_isLoaded = false;
  sim_bunchCrossing_isLoaded = false;
  see_stateCcov45_isLoaded = false;
  ph2_isStack_isLoaded = false;
  sim_trkShareFrac_isLoaded = false;
  trk_simTrkShareFrac_isLoaded = false;
  sim_phi_isLoaded = false;
  inv_side_isLoaded = false;
  vtx_fake_isLoaded = false;
  trk_nInactive_isLoaded = false;
  trk_nPixelLay_isLoaded = false;
  ph2_bbxi_isLoaded = false;
  vtx_xErr_isLoaded = false;
  see_stateCcov25_isLoaded = false;
  sim_parentVtxIdx_isLoaded = false;
  see_stateCcov23_isLoaded = false;
  trk_algoMask_isLoaded = false;
  trk_simTrkIdx_isLoaded = false;
  see_phiErr_isLoaded = false;
  trk_cotTheta_isLoaded = false;
  see_algo_isLoaded = false;
  simhit_module_isLoaded = false;
  simvtx_daughterSimIdx_isLoaded = false;
  vtx_x_isLoaded = false;
  trk_seedIdx_isLoaded = false;
  simhit_y_isLoaded = false;
  inv_layer_isLoaded = false;
  trk_nLostLay_isLoaded = false;
  ph2_isLower_isLoaded = false;
  pix_side_isLoaded = false;
  inv_isLower_isLoaded = false;
  ph2_trkIdx_isLoaded = false;
  sim_nValid_isLoaded = false;
  simhit_simTrkIdx_isLoaded = false;
  see_nCands_isLoaded = false;
  see_bestSimTrkIdx_isLoaded = false;
  vtx_yErr_isLoaded = false;
  trk_dzPV_isLoaded = false;
  ph2_xy_isLoaded = false;
  inv_module_isLoaded = false;
  see_stateCcov55_isLoaded = false;
  pix_panel_isLoaded = false;
  inv_ladder_isLoaded = false;
  ph2_xx_isLoaded = false;
  sim_pca_cotTheta_isLoaded = false;
  simpv_idx_isLoaded = false;
  trk_inner_pz_isLoaded = false;
  see_chi2_isLoaded = false;
  see_stateCcov35_isLoaded = false;
  see_stateCcov33_isLoaded = false;
  inv_detId_isLoaded = false;
  see_offset_isLoaded = false;
  sim_nLay_isLoaded = false;
  sim_simHitIdx_isLoaded = false;
  simhit_isUpper_isLoaded = false;
  see_stateCcov00_isLoaded = false;
  see_stopReason_isLoaded = false;
  vtx_valid_isLoaded = false;
  lumi_isLoaded = false;
  trk_refpoint_x_isLoaded = false;
  trk_refpoint_y_isLoaded = false;
  trk_refpoint_z_isLoaded = false;
  sim_n3DLay_isLoaded = false;
  see_nPhase2OT_isLoaded = false;
  trk_bestFromFirstHitSimTrkShareFracSimDenom_isLoaded = false;
  ph2_yy_isLoaded = false;
  ph2_yz_isLoaded = false;
  inv_blade_isLoaded = false;
  trk_ptErr_isLoaded = false;
  pix_zx_isLoaded = false;
  simvtx_z_isLoaded = false;
  sim_nTrackerHits_isLoaded = false;
  ph2_subdet_isLoaded = false;
  see_stateTrajPx_isLoaded = false;
  simhit_hitIdx_isLoaded = false;
  simhit_ladder_isLoaded = false;
  ph2_layer_isLoaded = false;
  see_phi_isLoaded = false;
  trk_nChi2_1Dmod_isLoaded = false;
  trk_inner_py_isLoaded = false;
  trk_inner_px_isLoaded = false;
  trk_dxyErr_isLoaded = false;
  sim_nPixelLay_isLoaded = false;
  see_nValid_isLoaded = false;
  trk_inner_pt_isLoaded = false;
  see_stateTrajGlbPy_isLoaded = false;
}
void Trktree::LoadAllBranches() {

  if (sim_etadiffs_branch != 0) sim_etadiffs(); // Added by Kasia
  if (sim_phidiffs_branch != 0) sim_phidiffs(); // Added by Kasia
  if (sim_rjet_branch != 0) sim_rjet(); // Added by Kasia
  if (sim_jet_eta_branch != 0) sim_jet_eta(); // Added by Kasia
  if (sim_jet_phi_branch != 0) sim_jet_phi(); // Added by Kasia
  if (sim_jet_pt_branch != 0) sim_jet_pt(); // Added by Kasia

  if (see_stateCcov01_branch != 0)
    see_stateCcov01();
  if (simhit_rod_branch != 0)
    simhit_rod();
  if (trk_phi_branch != 0)
    trk_phi();
  if (bsp_x_branch != 0)
    bsp_x();
  if (see_stateCcov05_branch != 0)
    see_stateCcov05();
  if (see_stateCcov04_branch != 0)
    see_stateCcov04();
  if (trk_dxyPV_branch != 0)
    trk_dxyPV();
  if (simhit_tof_branch != 0)
    simhit_tof();
  if (sim_event_branch != 0)
    sim_event();
  if (simhit_isStack_branch != 0)
    simhit_isStack();
  if (trk_dz_branch != 0)
    trk_dz();
  if (see_stateCcov03_branch != 0)
    see_stateCcov03();
  if (sim_eta_branch != 0)
    sim_eta();
  if (simvtx_processType_branch != 0)
    simvtx_processType();
  if (pix_radL_branch != 0)
    pix_radL();
  if (see_stateCcov02_branch != 0)
    see_stateCcov02();
  if (see_nGlued_branch != 0)
    see_nGlued();
  if (trk_bestSimTrkIdx_branch != 0)
    trk_bestSimTrkIdx();
  if (see_stateTrajGlbPz_branch != 0)
    see_stateTrajGlbPz();
  if (pix_yz_branch != 0)
    pix_yz();
  if (pix_yy_branch != 0)
    pix_yy();
  if (simhit_process_branch != 0)
    simhit_process();
  if (see_stateCcov34_branch != 0)
    see_stateCcov34();
  if (trk_nInnerLost_branch != 0)
    trk_nInnerLost();
  if (see_py_branch != 0)
    see_py();
  if (sim_trkIdx_branch != 0)
    sim_trkIdx();
  if (trk_nLost_branch != 0)
    trk_nLost();
  if (pix_isBarrel_branch != 0)
    pix_isBarrel();
  if (see_dxyErr_branch != 0)
    see_dxyErr();
  if (simhit_detId_branch != 0)
    simhit_detId();
  if (simhit_subdet_branch != 0)
    simhit_subdet();
  if (see_hitIdx_branch != 0)
    see_hitIdx();
  if (see_pt_branch != 0)
    see_pt();
  if (ph2_detId_branch != 0)
    ph2_detId();
  if (trk_nStripLay_branch != 0)
    trk_nStripLay();
  if (see_bestFromFirstHitSimTrkIdx_branch != 0)
    see_bestFromFirstHitSimTrkIdx();
  if (sim_pca_pt_branch != 0)
    sim_pca_pt();
  if (see_trkIdx_branch != 0)
    see_trkIdx();
  if (trk_nCluster_branch != 0)
    trk_nCluster();
  if (trk_bestFromFirstHitSimTrkNChi2_branch != 0)
    trk_bestFromFirstHitSimTrkNChi2();
  if (trk_isHP_branch != 0)
    trk_isHP();
  if (simhit_hitType_branch != 0)
    simhit_hitType();
  if (ph2_isUpper_branch != 0)
    ph2_isUpper();
  if (see_nStrip_branch != 0)
    see_nStrip();
  if (trk_bestSimTrkShareFracSimClusterDenom_branch != 0)
    trk_bestSimTrkShareFracSimClusterDenom();
  if (simhit_side_branch != 0)
    simhit_side();
  if (simhit_x_branch != 0)
    simhit_x();
  if (see_q_branch != 0)
    see_q();
  if (simhit_z_branch != 0)
    simhit_z();
  if (sim_pca_lambda_branch != 0)
    sim_pca_lambda();
  if (sim_q_branch != 0)
    sim_q();
  if (pix_bbxi_branch != 0)
    pix_bbxi();
  if (ph2_order_branch != 0)
    ph2_order();
  if (ph2_module_branch != 0)
    ph2_module();
  if (inv_order_branch != 0)
    inv_order();
  if (trk_dzErr_branch != 0)
    trk_dzErr();
  if (trk_nInnerInactive_branch != 0)
    trk_nInnerInactive();
  if (see_fitok_branch != 0)
    see_fitok();
  if (simhit_blade_branch != 0)
    simhit_blade();
  if (inv_subdet_branch != 0)
    inv_subdet();
  if (pix_blade_branch != 0)
    pix_blade();
  if (pix_xx_branch != 0)
    pix_xx();
  if (pix_xy_branch != 0)
    pix_xy();
  if (simhit_panel_branch != 0)
    simhit_panel();
  if (sim_pz_branch != 0)
    sim_pz();
  if (trk_dxy_branch != 0)
    trk_dxy();
  if (sim_px_branch != 0)
    sim_px();
  if (trk_lambda_branch != 0)
    trk_lambda();
  if (see_stateCcov12_branch != 0)
    see_stateCcov12();
  if (sim_pt_branch != 0)
    sim_pt();
  if (sim_py_branch != 0)
    sim_py();
  if (sim_decayVtxIdx_branch != 0)
    sim_decayVtxIdx();
  if (pix_detId_branch != 0)
    pix_detId();
  if (trk_eta_branch != 0)
    trk_eta();
  if (see_dxy_branch != 0)
    see_dxy();
  if (sim_isFromBHadron_branch != 0)
    sim_isFromBHadron();
  if (simhit_eloss_branch != 0)
    simhit_eloss();
  if (see_stateCcov11_branch != 0)
    see_stateCcov11();
  if (simhit_pz_branch != 0)
    simhit_pz();
  if (sim_pdgId_branch != 0)
    sim_pdgId();
  if (trk_stopReason_branch != 0)
    trk_stopReason();
  if (sim_pca_phi_branch != 0)
    sim_pca_phi();
  if (simhit_isLower_branch != 0)
    simhit_isLower();
  if (inv_ring_branch != 0)
    inv_ring();
  if (ph2_simHitIdx_branch != 0)
    ph2_simHitIdx();
  if (simhit_order_branch != 0)
    simhit_order();
  if (trk_dxyClosestPV_branch != 0)
    trk_dxyClosestPV();
  if (pix_z_branch != 0)
    pix_z();
  if (pix_y_branch != 0)
    pix_y();
  if (pix_x_branch != 0)
    pix_x();
  if (see_hitType_branch != 0)
    see_hitType();
  if (see_statePt_branch != 0)
    see_statePt();
  if (simvtx_sourceSimIdx_branch != 0)
    simvtx_sourceSimIdx();
  if (event_branch != 0)
    event();
  if (pix_module_branch != 0)
    pix_module();
  if (ph2_side_branch != 0)
    ph2_side();
  if (trk_bestSimTrkNChi2_branch != 0)
    trk_bestSimTrkNChi2();
  if (see_stateTrajPy_branch != 0)
    see_stateTrajPy();
  if (inv_type_branch != 0)
    inv_type();
  if (bsp_z_branch != 0)
    bsp_z();
  if (bsp_y_branch != 0)
    bsp_y();
  if (simhit_py_branch != 0)
    simhit_py();
  if (see_simTrkIdx_branch != 0)
    see_simTrkIdx();
  if (see_stateTrajGlbZ_branch != 0)
    see_stateTrajGlbZ();
  if (see_stateTrajGlbX_branch != 0)
    see_stateTrajGlbX();
  if (see_stateTrajGlbY_branch != 0)
    see_stateTrajGlbY();
  if (trk_originalAlgo_branch != 0)
    trk_originalAlgo();
  if (trk_nPixel_branch != 0)
    trk_nPixel();
  if (see_stateCcov14_branch != 0)
    see_stateCcov14();
  if (see_stateCcov15_branch != 0)
    see_stateCcov15();
  if (trk_phiErr_branch != 0)
    trk_phiErr();
  if (see_stateCcov13_branch != 0)
    see_stateCcov13();
  if (pix_chargeFraction_branch != 0)
    pix_chargeFraction();
  if (trk_q_branch != 0)
    trk_q();
  if (sim_seedIdx_branch != 0)
    sim_seedIdx();
  if (see_dzErr_branch != 0)
    see_dzErr();
  if (sim_nRecoClusters_branch != 0)
    sim_nRecoClusters();
  if (run_branch != 0)
    run();
  if (ph2_xySignificance_branch != 0)
    ph2_xySignificance();
  if (trk_nChi2_branch != 0)
    trk_nChi2();
  if (pix_layer_branch != 0)
    pix_layer();
  if (pix_xySignificance_branch != 0)
    pix_xySignificance();
  if (sim_pca_eta_branch != 0)
    sim_pca_eta();
  if (see_bestSimTrkShareFrac_branch != 0)
    see_bestSimTrkShareFrac();
  if (see_etaErr_branch != 0)
    see_etaErr();
  if (trk_bestSimTrkShareFracSimDenom_branch != 0)
    trk_bestSimTrkShareFracSimDenom();
  if (bsp_sigmaz_branch != 0)
    bsp_sigmaz();
  if (bsp_sigmay_branch != 0)
    bsp_sigmay();
  if (bsp_sigmax_branch != 0)
    bsp_sigmax();
  if (pix_ladder_branch != 0)
    pix_ladder();
  if (trk_qualityMask_branch != 0)
    trk_qualityMask();
  if (trk_ndof_branch != 0)
    trk_ndof();
  if (pix_subdet_branch != 0)
    pix_subdet();
  if (ph2_seeIdx_branch != 0)
    ph2_seeIdx();
  if (inv_isUpper_branch != 0)
    inv_isUpper();
  if (ph2_zx_branch != 0)
    ph2_zx();
  if (pix_trkIdx_branch != 0)
    pix_trkIdx();
  if (trk_nOuterLost_branch != 0)
    trk_nOuterLost();
  if (inv_panel_branch != 0)
    inv_panel();
  if (vtx_z_branch != 0)
    vtx_z();
  if (simhit_layer_branch != 0)
    simhit_layer();
  if (vtx_y_branch != 0)
    vtx_y();
  if (ph2_isBarrel_branch != 0)
    ph2_isBarrel();
  if (pix_seeIdx_branch != 0)
    pix_seeIdx();
  if (trk_bestFromFirstHitSimTrkIdx_branch != 0)
    trk_bestFromFirstHitSimTrkIdx();
  if (simhit_px_branch != 0)
    simhit_px();
  if (see_stateTrajX_branch != 0)
    see_stateTrajX();
  if (see_stateTrajY_branch != 0)
    see_stateTrajY();
  if (trk_nOuterInactive_branch != 0)
    trk_nOuterInactive();
  if (sim_pca_dxy_branch != 0)
    sim_pca_dxy();
  if (trk_algo_branch != 0)
    trk_algo();
  if (trk_hitType_branch != 0)
    trk_hitType();
  if (trk_bestFromFirstHitSimTrkShareFrac_branch != 0)
    trk_bestFromFirstHitSimTrkShareFrac();
  if (inv_isBarrel_branch != 0)
    inv_isBarrel();
  if (simvtx_event_branch != 0)
    simvtx_event();
  if (ph2_z_branch != 0)
    ph2_z();
  if (ph2_x_branch != 0)
    ph2_x();
  if (ph2_y_branch != 0)
    ph2_y();
  if (sim_genPdgIds_branch != 0)
    sim_genPdgIds();
  if (trk_mva_branch != 0)
    trk_mva();
  if (see_stateCcov24_branch != 0)
    see_stateCcov24();
  if (trk_dzClosestPV_branch != 0)
    trk_dzClosestPV();
  if (see_nCluster_branch != 0)
    see_nCluster();
  if (inv_rod_branch != 0)
    inv_rod();
  if (trk_hitIdx_branch != 0)
    trk_hitIdx();
  if (see_stateCcov22_branch != 0)
    see_stateCcov22();
  if (pix_simType_branch != 0)
    pix_simType();
  if (simhit_ring_branch != 0)
    simhit_ring();
  if (trk_outer_px_branch != 0)
    trk_outer_px();
  if (trk_outer_py_branch != 0)
    trk_outer_py();
  if (trk_outer_pz_branch != 0)
    trk_outer_pz();
  if (ph2_zz_branch != 0)
    ph2_zz();
  if (trk_outer_pt_branch != 0)
    trk_outer_pt();
  if (trk_n3DLay_branch != 0)
    trk_n3DLay();
  if (trk_nValid_branch != 0)
    trk_nValid();
  if (see_ptErr_branch != 0)
    see_ptErr();
  if (see_stateTrajGlbPx_branch != 0)
    see_stateTrajGlbPx();
  if (ph2_simType_branch != 0)
    ph2_simType();
  if (trk_bestFromFirstHitSimTrkShareFracSimClusterDenom_branch != 0)
    trk_bestFromFirstHitSimTrkShareFracSimClusterDenom();
  if (sim_hits_branch != 0)
    sim_hits();
  if (sim_len_branch != 0)
    sim_len();
  if (sim_lengap_branch != 0)
    sim_lengap();
  if (simvtx_x_branch != 0)
    simvtx_x();
  if (trk_pz_branch != 0)
    trk_pz();
  if (see_bestFromFirstHitSimTrkShareFrac_branch != 0)
    see_bestFromFirstHitSimTrkShareFrac();
  if (trk_px_branch != 0)
    trk_px();
  if (trk_py_branch != 0)
    trk_py();
  if (trk_vtxIdx_branch != 0)
    trk_vtxIdx();
  if (sim_nPixel_branch != 0)
    sim_nPixel();
  if (vtx_chi2_branch != 0)
    vtx_chi2();
  if (ph2_ring_branch != 0)
    ph2_ring();
  if (trk_pt_branch != 0)
    trk_pt();
  if (see_stateCcov44_branch != 0)
    see_stateCcov44();
  if (ph2_radL_branch != 0)
    ph2_radL();
  if (vtx_zErr_branch != 0)
    vtx_zErr();
  if (see_px_branch != 0)
    see_px();
  if (see_pz_branch != 0)
    see_pz();
  if (see_eta_branch != 0)
    see_eta();
  if (simvtx_bunchCrossing_branch != 0)
    simvtx_bunchCrossing();
  if (sim_pca_dz_branch != 0)
    sim_pca_dz();
  if (simvtx_y_branch != 0)
    simvtx_y();
  if (inv_isStack_branch != 0)
    inv_isStack();
  if (trk_nStrip_branch != 0)
    trk_nStrip();
  if (trk_etaErr_branch != 0)
    trk_etaErr();
  if (trk_simTrkNChi2_branch != 0)
    trk_simTrkNChi2();
  if (pix_zz_branch != 0)
    pix_zz();
  if (simhit_particle_branch != 0)
    simhit_particle();
  if (see_dz_branch != 0)
    see_dz();
  if (see_stateTrajPz_branch != 0)
    see_stateTrajPz();
  if (trk_bestSimTrkShareFrac_branch != 0)
    trk_bestSimTrkShareFrac();
  if (trk_lambdaErr_branch != 0)
    trk_lambdaErr();
  if (see_simTrkShareFrac_branch != 0)
    see_simTrkShareFrac();
  if (pix_simHitIdx_branch != 0)
    pix_simHitIdx();
  if (vtx_trkIdx_branch != 0)
    vtx_trkIdx();
  if (ph2_rod_branch != 0)
    ph2_rod();
  if (vtx_ndof_branch != 0)
    vtx_ndof();
  if (see_nPixel_branch != 0)
    see_nPixel();
  if (sim_nStrip_branch != 0)
    sim_nStrip();
  if (sim_bunchCrossing_branch != 0)
    sim_bunchCrossing();
  if (see_stateCcov45_branch != 0)
    see_stateCcov45();
  if (ph2_isStack_branch != 0)
    ph2_isStack();
  if (sim_trkShareFrac_branch != 0)
    sim_trkShareFrac();
  if (trk_simTrkShareFrac_branch != 0)
    trk_simTrkShareFrac();
  if (sim_phi_branch != 0)
    sim_phi();
  if (inv_side_branch != 0)
    inv_side();
  if (vtx_fake_branch != 0)
    vtx_fake();
  if (trk_nInactive_branch != 0)
    trk_nInactive();
  if (trk_nPixelLay_branch != 0)
    trk_nPixelLay();
  if (ph2_bbxi_branch != 0)
    ph2_bbxi();
  if (vtx_xErr_branch != 0)
    vtx_xErr();
  if (see_stateCcov25_branch != 0)
    see_stateCcov25();
  if (sim_parentVtxIdx_branch != 0)
    sim_parentVtxIdx();
  if (see_stateCcov23_branch != 0)
    see_stateCcov23();
  if (trk_algoMask_branch != 0)
    trk_algoMask();
  if (trk_simTrkIdx_branch != 0)
    trk_simTrkIdx();
  if (see_phiErr_branch != 0)
    see_phiErr();
  if (trk_cotTheta_branch != 0)
    trk_cotTheta();
  if (see_algo_branch != 0)
    see_algo();
  if (simhit_module_branch != 0)
    simhit_module();
  if (simvtx_daughterSimIdx_branch != 0)
    simvtx_daughterSimIdx();
  if (vtx_x_branch != 0)
    vtx_x();
  if (trk_seedIdx_branch != 0)
    trk_seedIdx();
  if (simhit_y_branch != 0)
    simhit_y();
  if (inv_layer_branch != 0)
    inv_layer();
  if (trk_nLostLay_branch != 0)
    trk_nLostLay();
  if (ph2_isLower_branch != 0)
    ph2_isLower();
  if (pix_side_branch != 0)
    pix_side();
  if (inv_isLower_branch != 0)
    inv_isLower();
  if (ph2_trkIdx_branch != 0)
    ph2_trkIdx();
  if (sim_nValid_branch != 0)
    sim_nValid();
  if (simhit_simTrkIdx_branch != 0)
    simhit_simTrkIdx();
  if (see_nCands_branch != 0)
    see_nCands();
  if (see_bestSimTrkIdx_branch != 0)
    see_bestSimTrkIdx();
  if (vtx_yErr_branch != 0)
    vtx_yErr();
  if (trk_dzPV_branch != 0)
    trk_dzPV();
  if (ph2_xy_branch != 0)
    ph2_xy();
  if (inv_module_branch != 0)
    inv_module();
  if (see_stateCcov55_branch != 0)
    see_stateCcov55();
  if (pix_panel_branch != 0)
    pix_panel();
  if (inv_ladder_branch != 0)
    inv_ladder();
  if (ph2_xx_branch != 0)
    ph2_xx();
  if (sim_pca_cotTheta_branch != 0)
    sim_pca_cotTheta();
  if (simpv_idx_branch != 0)
    simpv_idx();
  if (trk_inner_pz_branch != 0)
    trk_inner_pz();
  if (see_chi2_branch != 0)
    see_chi2();
  if (see_stateCcov35_branch != 0)
    see_stateCcov35();
  if (see_stateCcov33_branch != 0)
    see_stateCcov33();
  if (inv_detId_branch != 0)
    inv_detId();
  if (see_offset_branch != 0)
    see_offset();
  if (sim_nLay_branch != 0)
    sim_nLay();
  if (sim_simHitIdx_branch != 0)
    sim_simHitIdx();
  if (simhit_isUpper_branch != 0)
    simhit_isUpper();
  if (see_stateCcov00_branch != 0)
    see_stateCcov00();
  if (see_stopReason_branch != 0)
    see_stopReason();
  if (vtx_valid_branch != 0)
    vtx_valid();
  if (lumi_branch != 0)
    lumi();
  if (trk_refpoint_x_branch != 0)
    trk_refpoint_x();
  if (trk_refpoint_y_branch != 0)
    trk_refpoint_y();
  if (trk_refpoint_z_branch != 0)
    trk_refpoint_z();
  if (sim_n3DLay_branch != 0)
    sim_n3DLay();
  if (see_nPhase2OT_branch != 0)
    see_nPhase2OT();
  if (trk_bestFromFirstHitSimTrkShareFracSimDenom_branch != 0)
    trk_bestFromFirstHitSimTrkShareFracSimDenom();
  if (ph2_yy_branch != 0)
    ph2_yy();
  if (ph2_yz_branch != 0)
    ph2_yz();
  if (inv_blade_branch != 0)
    inv_blade();
  if (trk_ptErr_branch != 0)
    trk_ptErr();
  if (pix_zx_branch != 0)
    pix_zx();
  if (simvtx_z_branch != 0)
    simvtx_z();
  if (sim_nTrackerHits_branch != 0)
    sim_nTrackerHits();
  if (ph2_subdet_branch != 0)
    ph2_subdet();
  if (see_stateTrajPx_branch != 0)
    see_stateTrajPx();
  if (simhit_hitIdx_branch != 0)
    simhit_hitIdx();
  if (simhit_ladder_branch != 0)
    simhit_ladder();
  if (ph2_layer_branch != 0)
    ph2_layer();
  if (see_phi_branch != 0)
    see_phi();
  if (trk_nChi2_1Dmod_branch != 0)
    trk_nChi2_1Dmod();
  if (trk_inner_py_branch != 0)
    trk_inner_py();
  if (trk_inner_px_branch != 0)
    trk_inner_px();
  if (trk_dxyErr_branch != 0)
    trk_dxyErr();
  if (sim_nPixelLay_branch != 0)
    sim_nPixelLay();
  if (see_nValid_branch != 0)
    see_nValid();
  if (trk_inner_pt_branch != 0)
    trk_inner_pt();
  if (see_stateTrajGlbPy_branch != 0)
    see_stateTrajGlbPy();
}

// Added by Kasia
const std::vector<float> &Trktree::sim_etadiffs() {
  if (not sim_etadiffs_isLoaded) {
    if (sim_etadiffs_branch != 0) {
      sim_etadiffs_branch->GetEntry(index);
    } else {
      printf("branch sim_etadiffs_branch does not exist!\n");
      exit(1);
    }
    sim_etadiffs_isLoaded = true;
  }
  return *sim_etadiffs_;
}
// Added by Kasia
const std::vector<float> &Trktree::sim_phidiffs() {
  if (not sim_phidiffs_isLoaded) {
    if (sim_phidiffs_branch != 0) {
      sim_phidiffs_branch->GetEntry(index);
    } else {
      printf("branch sim_phidiffs_branch does not exist!\n");
      exit(1);
    }
    sim_phidiffs_isLoaded = true;
  }
  return *sim_phidiffs_;
}
// Added by Kasia
const std::vector<float> &Trktree::sim_rjet() {
  if (not sim_rjet_isLoaded) {
    if (sim_rjet_branch != 0) {
      sim_rjet_branch->GetEntry(index);
    } else {
      printf("branch sim_rjet_branch does not exist!\n");
      exit(1);
    }
    sim_rjet_isLoaded = true;
  }
  return *sim_rjet_;
}
// Added by Kasia
const std::vector<float> &Trktree::sim_jet_eta() {
  if (not sim_jet_eta_isLoaded) {
    if (sim_jet_eta_branch != 0) {
      sim_jet_eta_branch->GetEntry(index);
    } else {
      printf("branch sim_jet_eta_branch does not exist!\n");
      exit(1);
    }
    sim_jet_eta_isLoaded = true;
  }
  return *sim_jet_eta_;
}
// Added by Kasia
const std::vector<float> &Trktree::sim_jet_phi() {
  if (not sim_jet_phi_isLoaded) {
    if (sim_jet_phi_branch != 0) {
      sim_jet_phi_branch->GetEntry(index);
    } else {
      printf("branch sim_jet_phi_branch does not exist!\n");
      exit(1);
    }
    sim_jet_phi_isLoaded = true;
  }
  return *sim_jet_phi_;
}
// Added by Kasia
const std::vector<float> &Trktree::sim_jet_pt() {
  if (not sim_jet_pt_isLoaded) {
    if (sim_jet_pt_branch != 0) {
      sim_jet_pt_branch->GetEntry(index);
    } else {
      printf("branch sim_jet_pt_branch does not exist!\n");
      exit(1);
    }
    sim_jet_pt_isLoaded = true;
  }
  return *sim_jet_pt_;
}

const std::vector<float> &Trktree::see_stateCcov01() {
  if (not see_stateCcov01_isLoaded) {
    if (see_stateCcov01_branch != 0) {
      see_stateCcov01_branch->GetEntry(index);
    } else {
      printf("branch see_stateCcov01_branch does not exist!\n");
      exit(1);
    }
    see_stateCcov01_isLoaded = true;
  }
  return *see_stateCcov01_;
}
const std::vector<unsigned short> &Trktree::simhit_rod() {
  if (not simhit_rod_isLoaded) {
    if (simhit_rod_branch != 0) {
      simhit_rod_branch->GetEntry(index);
    } else {
      printf("branch simhit_rod_branch does not exist!\n");
      exit(1);
    }
    simhit_rod_isLoaded = true;
  }
  return *simhit_rod_;
}
const std::vector<float> &Trktree::trk_phi() {
  if (not trk_phi_isLoaded) {
    if (trk_phi_branch != 0) {
      trk_phi_branch->GetEntry(index);
    } else {
      printf("branch trk_phi_branch does not exist!\n");
      exit(1);
    }
    trk_phi_isLoaded = true;
  }
  return *trk_phi_;
}
const float &Trktree::bsp_x() {
  if (not bsp_x_isLoaded) {
    if (bsp_x_branch != 0) {
      bsp_x_branch->GetEntry(index);
    } else {
      printf("branch bsp_x_branch does not exist!\n");
      exit(1);
    }
    bsp_x_isLoaded = true;
  }
  return bsp_x_;
}
const std::vector<float> &Trktree::see_stateCcov05() {
  if (not see_stateCcov05_isLoaded) {
    if (see_stateCcov05_branch != 0) {
      see_stateCcov05_branch->GetEntry(index);
    } else {
      printf("branch see_stateCcov05_branch does not exist!\n");
      exit(1);
    }
    see_stateCcov05_isLoaded = true;
  }
  return *see_stateCcov05_;
}
const std::vector<float> &Trktree::see_stateCcov04() {
  if (not see_stateCcov04_isLoaded) {
    if (see_stateCcov04_branch != 0) {
      see_stateCcov04_branch->GetEntry(index);
    } else {
      printf("branch see_stateCcov04_branch does not exist!\n");
      exit(1);
    }
    see_stateCcov04_isLoaded = true;
  }
  return *see_stateCcov04_;
}
const std::vector<float> &Trktree::trk_dxyPV() {
  if (not trk_dxyPV_isLoaded) {
    if (trk_dxyPV_branch != 0) {
      trk_dxyPV_branch->GetEntry(index);
    } else {
      printf("branch trk_dxyPV_branch does not exist!\n");
      exit(1);
    }
    trk_dxyPV_isLoaded = true;
  }
  return *trk_dxyPV_;
}
const std::vector<float> &Trktree::simhit_tof() {
  if (not simhit_tof_isLoaded) {
    if (simhit_tof_branch != 0) {
      simhit_tof_branch->GetEntry(index);
    } else {
      printf("branch simhit_tof_branch does not exist!\n");
      exit(1);
    }
    simhit_tof_isLoaded = true;
  }
  return *simhit_tof_;
}
const std::vector<int> &Trktree::sim_event() {
  if (not sim_event_isLoaded) {
    if (sim_event_branch != 0) {
      sim_event_branch->GetEntry(index);
    } else {
      printf("branch sim_event_branch does not exist!\n");
      exit(1);
    }
    sim_event_isLoaded = true;
  }
  return *sim_event_;
}
const std::vector<unsigned short> &Trktree::simhit_isStack() {
  if (not simhit_isStack_isLoaded) {
    if (simhit_isStack_branch != 0) {
      simhit_isStack_branch->GetEntry(index);
    } else {
      printf("branch simhit_isStack_branch does not exist!\n");
      exit(1);
    }
    simhit_isStack_isLoaded = true;
  }
  return *simhit_isStack_;
}
const std::vector<float> &Trktree::trk_dz() {
  if (not trk_dz_isLoaded) {
    if (trk_dz_branch != 0) {
      trk_dz_branch->GetEntry(index);
    } else {
      printf("branch trk_dz_branch does not exist!\n");
      exit(1);
    }
    trk_dz_isLoaded = true;
  }
  return *trk_dz_;
}
const std::vector<float> &Trktree::see_stateCcov03() {
  if (not see_stateCcov03_isLoaded) {
    if (see_stateCcov03_branch != 0) {
      see_stateCcov03_branch->GetEntry(index);
    } else {
      printf("branch see_stateCcov03_branch does not exist!\n");
      exit(1);
    }
    see_stateCcov03_isLoaded = true;
  }
  return *see_stateCcov03_;
}
const std::vector<float> &Trktree::sim_eta() {
  if (not sim_eta_isLoaded) {
    if (sim_eta_branch != 0) {
      sim_eta_branch->GetEntry(index);
    } else {
      printf("branch sim_eta_branch does not exist!\n");
      exit(1);
    }
    sim_eta_isLoaded = true;
  }
  return *sim_eta_;
}
const std::vector<unsigned int> &Trktree::simvtx_processType() {
  if (not simvtx_processType_isLoaded) {
    if (simvtx_processType_branch != 0) {
      simvtx_processType_branch->GetEntry(index);
    } else {
      printf("branch simvtx_processType_branch does not exist!\n");
      exit(1);
    }
    simvtx_processType_isLoaded = true;
  }
  return *simvtx_processType_;
}
const std::vector<float> &Trktree::pix_radL() {
  if (not pix_radL_isLoaded) {
    if (pix_radL_branch != 0) {
      pix_radL_branch->GetEntry(index);
    } else {
      printf("branch pix_radL_branch does not exist!\n");
      exit(1);
    }
    pix_radL_isLoaded = true;
  }
  return *pix_radL_;
}
const std::vector<float> &Trktree::see_stateCcov02() {
  if (not see_stateCcov02_isLoaded) {
    if (see_stateCcov02_branch != 0) {
      see_stateCcov02_branch->GetEntry(index);
    } else {
      printf("branch see_stateCcov02_branch does not exist!\n");
      exit(1);
    }
    see_stateCcov02_isLoaded = true;
  }
  return *see_stateCcov02_;
}
const std::vector<unsigned int> &Trktree::see_nGlued() {
  if (not see_nGlued_isLoaded) {
    if (see_nGlued_branch != 0) {
      see_nGlued_branch->GetEntry(index);
    } else {
      printf("branch see_nGlued_branch does not exist!\n");
      exit(1);
    }
    see_nGlued_isLoaded = true;
  }
  return *see_nGlued_;
}
const std::vector<int> &Trktree::trk_bestSimTrkIdx() {
  if (not trk_bestSimTrkIdx_isLoaded) {
    if (trk_bestSimTrkIdx_branch != 0) {
      trk_bestSimTrkIdx_branch->GetEntry(index);
    } else {
      printf("branch trk_bestSimTrkIdx_branch does not exist!\n");
      exit(1);
    }
    trk_bestSimTrkIdx_isLoaded = true;
  }
  return *trk_bestSimTrkIdx_;
}
const std::vector<float> &Trktree::see_stateTrajGlbPz() {
  if (not see_stateTrajGlbPz_isLoaded) {
    if (see_stateTrajGlbPz_branch != 0) {
      see_stateTrajGlbPz_branch->GetEntry(index);
    } else {
      printf("branch see_stateTrajGlbPz_branch does not exist!\n");
      exit(1);
    }
    see_stateTrajGlbPz_isLoaded = true;
  }
  return *see_stateTrajGlbPz_;
}
const std::vector<float> &Trktree::pix_yz() {
  if (not pix_yz_isLoaded) {
    if (pix_yz_branch != 0) {
      pix_yz_branch->GetEntry(index);
    } else {
      printf("branch pix_yz_branch does not exist!\n");
      exit(1);
    }
    pix_yz_isLoaded = true;
  }
  return *pix_yz_;
}
const std::vector<float> &Trktree::pix_yy() {
  if (not pix_yy_isLoaded) {
    if (pix_yy_branch != 0) {
      pix_yy_branch->GetEntry(index);
    } else {
      printf("branch pix_yy_branch does not exist!\n");
      exit(1);
    }
    pix_yy_isLoaded = true;
  }
  return *pix_yy_;
}
const std::vector<short> &Trktree::simhit_process() {
  if (not simhit_process_isLoaded) {
    if (simhit_process_branch != 0) {
      simhit_process_branch->GetEntry(index);
    } else {
      printf("branch simhit_process_branch does not exist!\n");
      exit(1);
    }
    simhit_process_isLoaded = true;
  }
  return *simhit_process_;
}
const std::vector<float> &Trktree::see_stateCcov34() {
  if (not see_stateCcov34_isLoaded) {
    if (see_stateCcov34_branch != 0) {
      see_stateCcov34_branch->GetEntry(index);
    } else {
      printf("branch see_stateCcov34_branch does not exist!\n");
      exit(1);
    }
    see_stateCcov34_isLoaded = true;
  }
  return *see_stateCcov34_;
}
const std::vector<unsigned int> &Trktree::trk_nInnerLost() {
  if (not trk_nInnerLost_isLoaded) {
    if (trk_nInnerLost_branch != 0) {
      trk_nInnerLost_branch->GetEntry(index);
    } else {
      printf("branch trk_nInnerLost_branch does not exist!\n");
      exit(1);
    }
    trk_nInnerLost_isLoaded = true;
  }
  return *trk_nInnerLost_;
}
const std::vector<float> &Trktree::see_py() {
  if (not see_py_isLoaded) {
    if (see_py_branch != 0) {
      see_py_branch->GetEntry(index);
    } else {
      printf("branch see_py_branch does not exist!\n");
      exit(1);
    }
    see_py_isLoaded = true;
  }
  return *see_py_;
}
const std::vector<std::vector<int> > &Trktree::sim_trkIdx() {
  if (not sim_trkIdx_isLoaded) {
    if (sim_trkIdx_branch != 0) {
      sim_trkIdx_branch->GetEntry(index);
    } else {
      printf("branch sim_trkIdx_branch does not exist!\n");
      exit(1);
    }
    sim_trkIdx_isLoaded = true;
  }
  return *sim_trkIdx_;
}
const std::vector<unsigned int> &Trktree::trk_nLost() {
  if (not trk_nLost_isLoaded) {
    if (trk_nLost_branch != 0) {
      trk_nLost_branch->GetEntry(index);
    } else {
      printf("branch trk_nLost_branch does not exist!\n");
      exit(1);
    }
    trk_nLost_isLoaded = true;
  }
  return *trk_nLost_;
}
const std::vector<short> &Trktree::pix_isBarrel() {
  if (not pix_isBarrel_isLoaded) {
    if (pix_isBarrel_branch != 0) {
      pix_isBarrel_branch->GetEntry(index);
    } else {
      printf("branch pix_isBarrel_branch does not exist!\n");
      exit(1);
    }
    pix_isBarrel_isLoaded = true;
  }
  return *pix_isBarrel_;
}
const std::vector<float> &Trktree::see_dxyErr() {
  if (not see_dxyErr_isLoaded) {
    if (see_dxyErr_branch != 0) {
      see_dxyErr_branch->GetEntry(index);
    } else {
      printf("branch see_dxyErr_branch does not exist!\n");
      exit(1);
    }
    see_dxyErr_isLoaded = true;
  }
  return *see_dxyErr_;
}
const std::vector<unsigned int> &Trktree::simhit_detId() {
  if (not simhit_detId_isLoaded) {
    if (simhit_detId_branch != 0) {
      simhit_detId_branch->GetEntry(index);
    } else {
      printf("branch simhit_detId_branch does not exist!\n");
      exit(1);
    }
    simhit_detId_isLoaded = true;
  }
  return *simhit_detId_;
}
const std::vector<unsigned short> &Trktree::simhit_subdet() {
  if (not simhit_subdet_isLoaded) {
    if (simhit_subdet_branch != 0) {
      simhit_subdet_branch->GetEntry(index);
    } else {
      printf("branch simhit_subdet_branch does not exist!\n");
      exit(1);
    }
    simhit_subdet_isLoaded = true;
  }
  return *simhit_subdet_;
}
const std::vector<std::vector<int> > &Trktree::see_hitIdx() {
  if (not see_hitIdx_isLoaded) {
    if (see_hitIdx_branch != 0) {
      see_hitIdx_branch->GetEntry(index);
    } else {
      printf("branch see_hitIdx_branch does not exist!\n");
      exit(1);
    }
    see_hitIdx_isLoaded = true;
  }
  return *see_hitIdx_;
}
const std::vector<float> &Trktree::see_pt() {
  if (not see_pt_isLoaded) {
    if (see_pt_branch != 0) {
      see_pt_branch->GetEntry(index);
    } else {
      printf("branch see_pt_branch does not exist!\n");
      exit(1);
    }
    see_pt_isLoaded = true;
  }
  return *see_pt_;
}
const std::vector<unsigned int> &Trktree::ph2_detId() {
  if (not ph2_detId_isLoaded) {
    if (ph2_detId_branch != 0) {
      ph2_detId_branch->GetEntry(index);
    } else {
      printf("branch ph2_detId_branch does not exist!\n");
      exit(1);
    }
    ph2_detId_isLoaded = true;
  }
  return *ph2_detId_;
}
const std::vector<unsigned int> &Trktree::trk_nStripLay() {
  if (not trk_nStripLay_isLoaded) {
    if (trk_nStripLay_branch != 0) {
      trk_nStripLay_branch->GetEntry(index);
    } else {
      printf("branch trk_nStripLay_branch does not exist!\n");
      exit(1);
    }
    trk_nStripLay_isLoaded = true;
  }
  return *trk_nStripLay_;
}
const std::vector<int> &Trktree::see_bestFromFirstHitSimTrkIdx() {
  if (not see_bestFromFirstHitSimTrkIdx_isLoaded) {
    if (see_bestFromFirstHitSimTrkIdx_branch != 0) {
      see_bestFromFirstHitSimTrkIdx_branch->GetEntry(index);
    } else {
      printf("branch see_bestFromFirstHitSimTrkIdx_branch does not exist!\n");
      exit(1);
    }
    see_bestFromFirstHitSimTrkIdx_isLoaded = true;
  }
  return *see_bestFromFirstHitSimTrkIdx_;
}
const std::vector<float> &Trktree::sim_pca_pt() {
  if (not sim_pca_pt_isLoaded) {
    if (sim_pca_pt_branch != 0) {
      sim_pca_pt_branch->GetEntry(index);
    } else {
      printf("branch sim_pca_pt_branch does not exist!\n");
      exit(1);
    }
    sim_pca_pt_isLoaded = true;
  }
  return *sim_pca_pt_;
}
const std::vector<int> &Trktree::see_trkIdx() {
  if (not see_trkIdx_isLoaded) {
    if (see_trkIdx_branch != 0) {
      see_trkIdx_branch->GetEntry(index);
    } else {
      printf("branch see_trkIdx_branch does not exist!\n");
      exit(1);
    }
    see_trkIdx_isLoaded = true;
  }
  return *see_trkIdx_;
}
const std::vector<unsigned int> &Trktree::trk_nCluster() {
  if (not trk_nCluster_isLoaded) {
    if (trk_nCluster_branch != 0) {
      trk_nCluster_branch->GetEntry(index);
    } else {
      printf("branch trk_nCluster_branch does not exist!\n");
      exit(1);
    }
    trk_nCluster_isLoaded = true;
  }
  return *trk_nCluster_;
}
const std::vector<float> &Trktree::trk_bestFromFirstHitSimTrkNChi2() {
  if (not trk_bestFromFirstHitSimTrkNChi2_isLoaded) {
    if (trk_bestFromFirstHitSimTrkNChi2_branch != 0) {
      trk_bestFromFirstHitSimTrkNChi2_branch->GetEntry(index);
    } else {
      printf("branch trk_bestFromFirstHitSimTrkNChi2_branch does not exist!\n");
      exit(1);
    }
    trk_bestFromFirstHitSimTrkNChi2_isLoaded = true;
  }
  return *trk_bestFromFirstHitSimTrkNChi2_;
}
const std::vector<short> &Trktree::trk_isHP() {
  if (not trk_isHP_isLoaded) {
    if (trk_isHP_branch != 0) {
      trk_isHP_branch->GetEntry(index);
    } else {
      printf("branch trk_isHP_branch does not exist!\n");
      exit(1);
    }
    trk_isHP_isLoaded = true;
  }
  return *trk_isHP_;
}
const std::vector<std::vector<int> > &Trktree::simhit_hitType() {
  if (not simhit_hitType_isLoaded) {
    if (simhit_hitType_branch != 0) {
      simhit_hitType_branch->GetEntry(index);
    } else {
      printf("branch simhit_hitType_branch does not exist!\n");
      exit(1);
    }
    simhit_hitType_isLoaded = true;
  }
  return *simhit_hitType_;
}
const std::vector<unsigned short> &Trktree::ph2_isUpper() {
  if (not ph2_isUpper_isLoaded) {
    if (ph2_isUpper_branch != 0) {
      ph2_isUpper_branch->GetEntry(index);
    } else {
      printf("branch ph2_isUpper_branch does not exist!\n");
      exit(1);
    }
    ph2_isUpper_isLoaded = true;
  }
  return *ph2_isUpper_;
}
const std::vector<unsigned int> &Trktree::see_nStrip() {
  if (not see_nStrip_isLoaded) {
    if (see_nStrip_branch != 0) {
      see_nStrip_branch->GetEntry(index);
    } else {
      printf("branch see_nStrip_branch does not exist!\n");
      exit(1);
    }
    see_nStrip_isLoaded = true;
  }
  return *see_nStrip_;
}
const std::vector<float> &Trktree::trk_bestSimTrkShareFracSimClusterDenom() {
  if (not trk_bestSimTrkShareFracSimClusterDenom_isLoaded) {
    if (trk_bestSimTrkShareFracSimClusterDenom_branch != 0) {
      trk_bestSimTrkShareFracSimClusterDenom_branch->GetEntry(index);
    } else {
      printf("branch trk_bestSimTrkShareFracSimClusterDenom_branch does not exist!\n");
      exit(1);
    }
    trk_bestSimTrkShareFracSimClusterDenom_isLoaded = true;
  }
  return *trk_bestSimTrkShareFracSimClusterDenom_;
}
const std::vector<unsigned short> &Trktree::simhit_side() {
  if (not simhit_side_isLoaded) {
    if (simhit_side_branch != 0) {
      simhit_side_branch->GetEntry(index);
    } else {
      printf("branch simhit_side_branch does not exist!\n");
      exit(1);
    }
    simhit_side_isLoaded = true;
  }
  return *simhit_side_;
}
const std::vector<float> &Trktree::simhit_x() {
  if (not simhit_x_isLoaded) {
    if (simhit_x_branch != 0) {
      simhit_x_branch->GetEntry(index);
    } else {
      printf("branch simhit_x_branch does not exist!\n");
      exit(1);
    }
    simhit_x_isLoaded = true;
  }
  return *simhit_x_;
}
const std::vector<int> &Trktree::see_q() {
  if (not see_q_isLoaded) {
    if (see_q_branch != 0) {
      see_q_branch->GetEntry(index);
    } else {
      printf("branch see_q_branch does not exist!\n");
      exit(1);
    }
    see_q_isLoaded = true;
  }
  return *see_q_;
}
const std::vector<float> &Trktree::simhit_z() {
  if (not simhit_z_isLoaded) {
    if (simhit_z_branch != 0) {
      simhit_z_branch->GetEntry(index);
    } else {
      printf("branch simhit_z_branch does not exist!\n");
      exit(1);
    }
    simhit_z_isLoaded = true;
  }
  return *simhit_z_;
}
const std::vector<float> &Trktree::sim_pca_lambda() {
  if (not sim_pca_lambda_isLoaded) {
    if (sim_pca_lambda_branch != 0) {
      sim_pca_lambda_branch->GetEntry(index);
    } else {
      printf("branch sim_pca_lambda_branch does not exist!\n");
      exit(1);
    }
    sim_pca_lambda_isLoaded = true;
  }
  return *sim_pca_lambda_;
}
const std::vector<int> &Trktree::sim_q() {
  if (not sim_q_isLoaded) {
    if (sim_q_branch != 0) {
      sim_q_branch->GetEntry(index);
    } else {
      printf("branch sim_q_branch does not exist!\n");
      exit(1);
    }
    sim_q_isLoaded = true;
  }
  return *sim_q_;
}
const std::vector<float> &Trktree::pix_bbxi() {
  if (not pix_bbxi_isLoaded) {
    if (pix_bbxi_branch != 0) {
      pix_bbxi_branch->GetEntry(index);
    } else {
      printf("branch pix_bbxi_branch does not exist!\n");
      exit(1);
    }
    pix_bbxi_isLoaded = true;
  }
  return *pix_bbxi_;
}
const std::vector<unsigned short> &Trktree::ph2_order() {
  if (not ph2_order_isLoaded) {
    if (ph2_order_branch != 0) {
      ph2_order_branch->GetEntry(index);
    } else {
      printf("branch ph2_order_branch does not exist!\n");
      exit(1);
    }
    ph2_order_isLoaded = true;
  }
  return *ph2_order_;
}
const std::vector<unsigned short> &Trktree::ph2_module() {
  if (not ph2_module_isLoaded) {
    if (ph2_module_branch != 0) {
      ph2_module_branch->GetEntry(index);
    } else {
      printf("branch ph2_module_branch does not exist!\n");
      exit(1);
    }
    ph2_module_isLoaded = true;
  }
  return *ph2_module_;
}
const std::vector<unsigned short> &Trktree::inv_order() {
  if (not inv_order_isLoaded) {
    if (inv_order_branch != 0) {
      inv_order_branch->GetEntry(index);
    } else {
      printf("branch inv_order_branch does not exist!\n");
      exit(1);
    }
    inv_order_isLoaded = true;
  }
  return *inv_order_;
}
const std::vector<float> &Trktree::trk_dzErr() {
  if (not trk_dzErr_isLoaded) {
    if (trk_dzErr_branch != 0) {
      trk_dzErr_branch->GetEntry(index);
    } else {
      printf("branch trk_dzErr_branch does not exist!\n");
      exit(1);
    }
    trk_dzErr_isLoaded = true;
  }
  return *trk_dzErr_;
}
const std::vector<unsigned int> &Trktree::trk_nInnerInactive() {
  if (not trk_nInnerInactive_isLoaded) {
    if (trk_nInnerInactive_branch != 0) {
      trk_nInnerInactive_branch->GetEntry(index);
    } else {
      printf("branch trk_nInnerInactive_branch does not exist!\n");
      exit(1);
    }
    trk_nInnerInactive_isLoaded = true;
  }
  return *trk_nInnerInactive_;
}
const std::vector<short> &Trktree::see_fitok() {
  if (not see_fitok_isLoaded) {
    if (see_fitok_branch != 0) {
      see_fitok_branch->GetEntry(index);
    } else {
      printf("branch see_fitok_branch does not exist!\n");
      exit(1);
    }
    see_fitok_isLoaded = true;
  }
  return *see_fitok_;
}
const std::vector<unsigned short> &Trktree::simhit_blade() {
  if (not simhit_blade_isLoaded) {
    if (simhit_blade_branch != 0) {
      simhit_blade_branch->GetEntry(index);
    } else {
      printf("branch simhit_blade_branch does not exist!\n");
      exit(1);
    }
    simhit_blade_isLoaded = true;
  }
  return *simhit_blade_;
}
const std::vector<unsigned short> &Trktree::inv_subdet() {
  if (not inv_subdet_isLoaded) {
    if (inv_subdet_branch != 0) {
      inv_subdet_branch->GetEntry(index);
    } else {
      printf("branch inv_subdet_branch does not exist!\n");
      exit(1);
    }
    inv_subdet_isLoaded = true;
  }
  return *inv_subdet_;
}
const std::vector<unsigned short> &Trktree::pix_blade() {
  if (not pix_blade_isLoaded) {
    if (pix_blade_branch != 0) {
      pix_blade_branch->GetEntry(index);
    } else {
      printf("branch pix_blade_branch does not exist!\n");
      exit(1);
    }
    pix_blade_isLoaded = true;
  }
  return *pix_blade_;
}
const std::vector<float> &Trktree::pix_xx() {
  if (not pix_xx_isLoaded) {
    if (pix_xx_branch != 0) {
      pix_xx_branch->GetEntry(index);
    } else {
      printf("branch pix_xx_branch does not exist!\n");
      exit(1);
    }
    pix_xx_isLoaded = true;
  }
  return *pix_xx_;
}
const std::vector<float> &Trktree::pix_xy() {
  if (not pix_xy_isLoaded) {
    if (pix_xy_branch != 0) {
      pix_xy_branch->GetEntry(index);
    } else {
      printf("branch pix_xy_branch does not exist!\n");
      exit(1);
    }
    pix_xy_isLoaded = true;
  }
  return *pix_xy_;
}
const std::vector<unsigned short> &Trktree::simhit_panel() {
  if (not simhit_panel_isLoaded) {
    if (simhit_panel_branch != 0) {
      simhit_panel_branch->GetEntry(index);
    } else {
      printf("branch simhit_panel_branch does not exist!\n");
      exit(1);
    }
    simhit_panel_isLoaded = true;
  }
  return *simhit_panel_;
}
const std::vector<float> &Trktree::sim_pz() {
  if (not sim_pz_isLoaded) {
    if (sim_pz_branch != 0) {
      sim_pz_branch->GetEntry(index);
    } else {
      printf("branch sim_pz_branch does not exist!\n");
      exit(1);
    }
    sim_pz_isLoaded = true;
  }
  return *sim_pz_;
}
const std::vector<float> &Trktree::trk_dxy() {
  if (not trk_dxy_isLoaded) {
    if (trk_dxy_branch != 0) {
      trk_dxy_branch->GetEntry(index);
    } else {
      printf("branch trk_dxy_branch does not exist!\n");
      exit(1);
    }
    trk_dxy_isLoaded = true;
  }
  return *trk_dxy_;
}
const std::vector<float> &Trktree::sim_px() {
  if (not sim_px_isLoaded) {
    if (sim_px_branch != 0) {
      sim_px_branch->GetEntry(index);
    } else {
      printf("branch sim_px_branch does not exist!\n");
      exit(1);
    }
    sim_px_isLoaded = true;
  }
  return *sim_px_;
}
const std::vector<float> &Trktree::trk_lambda() {
  if (not trk_lambda_isLoaded) {
    if (trk_lambda_branch != 0) {
      trk_lambda_branch->GetEntry(index);
    } else {
      printf("branch trk_lambda_branch does not exist!\n");
      exit(1);
    }
    trk_lambda_isLoaded = true;
  }
  return *trk_lambda_;
}
const std::vector<float> &Trktree::see_stateCcov12() {
  if (not see_stateCcov12_isLoaded) {
    if (see_stateCcov12_branch != 0) {
      see_stateCcov12_branch->GetEntry(index);
    } else {
      printf("branch see_stateCcov12_branch does not exist!\n");
      exit(1);
    }
    see_stateCcov12_isLoaded = true;
  }
  return *see_stateCcov12_;
}
const std::vector<float> &Trktree::sim_pt() {
  if (not sim_pt_isLoaded) {
    if (sim_pt_branch != 0) {
      sim_pt_branch->GetEntry(index);
    } else {
      printf("branch sim_pt_branch does not exist!\n");
      exit(1);
    }
    sim_pt_isLoaded = true;
  }
  return *sim_pt_;
}
const std::vector<float> &Trktree::sim_py() {
  if (not sim_py_isLoaded) {
    if (sim_py_branch != 0) {
      sim_py_branch->GetEntry(index);
    } else {
      printf("branch sim_py_branch does not exist!\n");
      exit(1);
    }
    sim_py_isLoaded = true;
  }
  return *sim_py_;
}
const std::vector<std::vector<int> > &Trktree::sim_decayVtxIdx() {
  if (not sim_decayVtxIdx_isLoaded) {
    if (sim_decayVtxIdx_branch != 0) {
      sim_decayVtxIdx_branch->GetEntry(index);
    } else {
      printf("branch sim_decayVtxIdx_branch does not exist!\n");
      exit(1);
    }
    sim_decayVtxIdx_isLoaded = true;
  }
  return *sim_decayVtxIdx_;
}
const std::vector<unsigned int> &Trktree::pix_detId() {
  if (not pix_detId_isLoaded) {
    if (pix_detId_branch != 0) {
      pix_detId_branch->GetEntry(index);
    } else {
      printf("branch pix_detId_branch does not exist!\n");
      exit(1);
    }
    pix_detId_isLoaded = true;
  }
  return *pix_detId_;
}
const std::vector<float> &Trktree::trk_eta() {
  if (not trk_eta_isLoaded) {
    if (trk_eta_branch != 0) {
      trk_eta_branch->GetEntry(index);
    } else {
      printf("branch trk_eta_branch does not exist!\n");
      exit(1);
    }
    trk_eta_isLoaded = true;
  }
  return *trk_eta_;
}
const std::vector<float> &Trktree::see_dxy() {
  if (not see_dxy_isLoaded) {
    if (see_dxy_branch != 0) {
      see_dxy_branch->GetEntry(index);
    } else {
      printf("branch see_dxy_branch does not exist!\n");
      exit(1);
    }
    see_dxy_isLoaded = true;
  }
  return *see_dxy_;
}
const std::vector<int> &Trktree::sim_isFromBHadron() {
  if (not sim_isFromBHadron_isLoaded) {
    if (sim_isFromBHadron_branch != 0) {
      sim_isFromBHadron_branch->GetEntry(index);
    } else {
      printf("branch sim_isFromBHadron_branch does not exist!\n");
      exit(1);
    }
    sim_isFromBHadron_isLoaded = true;
  }
  return *sim_isFromBHadron_;
}
const std::vector<float> &Trktree::simhit_eloss() {
  if (not simhit_eloss_isLoaded) {
    if (simhit_eloss_branch != 0) {
      simhit_eloss_branch->GetEntry(index);
    } else {
      printf("branch simhit_eloss_branch does not exist!\n");
      exit(1);
    }
    simhit_eloss_isLoaded = true;
  }
  return *simhit_eloss_;
}
const std::vector<float> &Trktree::see_stateCcov11() {
  if (not see_stateCcov11_isLoaded) {
    if (see_stateCcov11_branch != 0) {
      see_stateCcov11_branch->GetEntry(index);
    } else {
      printf("branch see_stateCcov11_branch does not exist!\n");
      exit(1);
    }
    see_stateCcov11_isLoaded = true;
  }
  return *see_stateCcov11_;
}
const std::vector<float> &Trktree::simhit_pz() {
  if (not simhit_pz_isLoaded) {
    if (simhit_pz_branch != 0) {
      simhit_pz_branch->GetEntry(index);
    } else {
      printf("branch simhit_pz_branch does not exist!\n");
      exit(1);
    }
    simhit_pz_isLoaded = true;
  }
  return *simhit_pz_;
}
const std::vector<int> &Trktree::sim_pdgId() {
  if (not sim_pdgId_isLoaded) {
    if (sim_pdgId_branch != 0) {
      sim_pdgId_branch->GetEntry(index);
    } else {
      printf("branch sim_pdgId_branch does not exist!\n");
      exit(1);
    }
    sim_pdgId_isLoaded = true;
  }
  return *sim_pdgId_;
}
const std::vector<unsigned short> &Trktree::trk_stopReason() {
  if (not trk_stopReason_isLoaded) {
    if (trk_stopReason_branch != 0) {
      trk_stopReason_branch->GetEntry(index);
    } else {
      printf("branch trk_stopReason_branch does not exist!\n");
      exit(1);
    }
    trk_stopReason_isLoaded = true;
  }
  return *trk_stopReason_;
}
const std::vector<float> &Trktree::sim_pca_phi() {
  if (not sim_pca_phi_isLoaded) {
    if (sim_pca_phi_branch != 0) {
      sim_pca_phi_branch->GetEntry(index);
    } else {
      printf("branch sim_pca_phi_branch does not exist!\n");
      exit(1);
    }
    sim_pca_phi_isLoaded = true;
  }
  return *sim_pca_phi_;
}
const std::vector<unsigned short> &Trktree::simhit_isLower() {
  if (not simhit_isLower_isLoaded) {
    if (simhit_isLower_branch != 0) {
      simhit_isLower_branch->GetEntry(index);
    } else {
      printf("branch simhit_isLower_branch does not exist!\n");
      exit(1);
    }
    simhit_isLower_isLoaded = true;
  }
  return *simhit_isLower_;
}
const std::vector<unsigned short> &Trktree::inv_ring() {
  if (not inv_ring_isLoaded) {
    if (inv_ring_branch != 0) {
      inv_ring_branch->GetEntry(index);
    } else {
      printf("branch inv_ring_branch does not exist!\n");
      exit(1);
    }
    inv_ring_isLoaded = true;
  }
  return *inv_ring_;
}
const std::vector<std::vector<int> > &Trktree::ph2_simHitIdx() {
  if (not ph2_simHitIdx_isLoaded) {
    if (ph2_simHitIdx_branch != 0) {
      ph2_simHitIdx_branch->GetEntry(index);
    } else {
      printf("branch ph2_simHitIdx_branch does not exist!\n");
      exit(1);
    }
    ph2_simHitIdx_isLoaded = true;
  }
  return *ph2_simHitIdx_;
}
const std::vector<unsigned short> &Trktree::simhit_order() {
  if (not simhit_order_isLoaded) {
    if (simhit_order_branch != 0) {
      simhit_order_branch->GetEntry(index);
    } else {
      printf("branch simhit_order_branch does not exist!\n");
      exit(1);
    }
    simhit_order_isLoaded = true;
  }
  return *simhit_order_;
}
const std::vector<float> &Trktree::trk_dxyClosestPV() {
  if (not trk_dxyClosestPV_isLoaded) {
    if (trk_dxyClosestPV_branch != 0) {
      trk_dxyClosestPV_branch->GetEntry(index);
    } else {
      printf("branch trk_dxyClosestPV_branch does not exist!\n");
      exit(1);
    }
    trk_dxyClosestPV_isLoaded = true;
  }
  return *trk_dxyClosestPV_;
}
const std::vector<float> &Trktree::pix_z() {
  if (not pix_z_isLoaded) {
    if (pix_z_branch != 0) {
      pix_z_branch->GetEntry(index);
    } else {
      printf("branch pix_z_branch does not exist!\n");
      exit(1);
    }
    pix_z_isLoaded = true;
  }
  return *pix_z_;
}
const std::vector<float> &Trktree::pix_y() {
  if (not pix_y_isLoaded) {
    if (pix_y_branch != 0) {
      pix_y_branch->GetEntry(index);
    } else {
      printf("branch pix_y_branch does not exist!\n");
      exit(1);
    }
    pix_y_isLoaded = true;
  }
  return *pix_y_;
}
const std::vector<float> &Trktree::pix_x() {
  if (not pix_x_isLoaded) {
    if (pix_x_branch != 0) {
      pix_x_branch->GetEntry(index);
    } else {
      printf("branch pix_x_branch does not exist!\n");
      exit(1);
    }
    pix_x_isLoaded = true;
  }
  return *pix_x_;
}
const std::vector<std::vector<int> > &Trktree::see_hitType() {
  if (not see_hitType_isLoaded) {
    if (see_hitType_branch != 0) {
      see_hitType_branch->GetEntry(index);
    } else {
      printf("branch see_hitType_branch does not exist!\n");
      exit(1);
    }
    see_hitType_isLoaded = true;
  }
  return *see_hitType_;
}
const std::vector<float> &Trktree::see_statePt() {
  if (not see_statePt_isLoaded) {
    if (see_statePt_branch != 0) {
      see_statePt_branch->GetEntry(index);
    } else {
      printf("branch see_statePt_branch does not exist!\n");
      exit(1);
    }
    see_statePt_isLoaded = true;
  }
  return *see_statePt_;
}
const std::vector<std::vector<int> > &Trktree::simvtx_sourceSimIdx() {
  if (not simvtx_sourceSimIdx_isLoaded) {
    if (simvtx_sourceSimIdx_branch != 0) {
      simvtx_sourceSimIdx_branch->GetEntry(index);
    } else {
      printf("branch simvtx_sourceSimIdx_branch does not exist!\n");
      exit(1);
    }
    simvtx_sourceSimIdx_isLoaded = true;
  }
  return *simvtx_sourceSimIdx_;
}
const unsigned long long &Trktree::event() {
  if (not event_isLoaded) {
    if (event_branch != 0) {
      event_branch->GetEntry(index);
    } else {
      printf("branch event_branch does not exist!\n");
      exit(1);
    }
    event_isLoaded = true;
  }
  return event_;
}
const std::vector<unsigned short> &Trktree::pix_module() {
  if (not pix_module_isLoaded) {
    if (pix_module_branch != 0) {
      pix_module_branch->GetEntry(index);
    } else {
      printf("branch pix_module_branch does not exist!\n");
      exit(1);
    }
    pix_module_isLoaded = true;
  }
  return *pix_module_;
}
const std::vector<unsigned short> &Trktree::ph2_side() {
  if (not ph2_side_isLoaded) {
    if (ph2_side_branch != 0) {
      ph2_side_branch->GetEntry(index);
    } else {
      printf("branch ph2_side_branch does not exist!\n");
      exit(1);
    }
    ph2_side_isLoaded = true;
  }
  return *ph2_side_;
}
const std::vector<float> &Trktree::trk_bestSimTrkNChi2() {
  if (not trk_bestSimTrkNChi2_isLoaded) {
    if (trk_bestSimTrkNChi2_branch != 0) {
      trk_bestSimTrkNChi2_branch->GetEntry(index);
    } else {
      printf("branch trk_bestSimTrkNChi2_branch does not exist!\n");
      exit(1);
    }
    trk_bestSimTrkNChi2_isLoaded = true;
  }
  return *trk_bestSimTrkNChi2_;
}
const std::vector<float> &Trktree::see_stateTrajPy() {
  if (not see_stateTrajPy_isLoaded) {
    if (see_stateTrajPy_branch != 0) {
      see_stateTrajPy_branch->GetEntry(index);
    } else {
      printf("branch see_stateTrajPy_branch does not exist!\n");
      exit(1);
    }
    see_stateTrajPy_isLoaded = true;
  }
  return *see_stateTrajPy_;
}
const std::vector<unsigned short> &Trktree::inv_type() {
  if (not inv_type_isLoaded) {
    if (inv_type_branch != 0) {
      inv_type_branch->GetEntry(index);
    } else {
      printf("branch inv_type_branch does not exist!\n");
      exit(1);
    }
    inv_type_isLoaded = true;
  }
  return *inv_type_;
}
const float &Trktree::bsp_z() {
  if (not bsp_z_isLoaded) {
    if (bsp_z_branch != 0) {
      bsp_z_branch->GetEntry(index);
    } else {
      printf("branch bsp_z_branch does not exist!\n");
      exit(1);
    }
    bsp_z_isLoaded = true;
  }
  return bsp_z_;
}
const float &Trktree::bsp_y() {
  if (not bsp_y_isLoaded) {
    if (bsp_y_branch != 0) {
      bsp_y_branch->GetEntry(index);
    } else {
      printf("branch bsp_y_branch does not exist!\n");
      exit(1);
    }
    bsp_y_isLoaded = true;
  }
  return bsp_y_;
}
const std::vector<float> &Trktree::simhit_py() {
  if (not simhit_py_isLoaded) {
    if (simhit_py_branch != 0) {
      simhit_py_branch->GetEntry(index);
    } else {
      printf("branch simhit_py_branch does not exist!\n");
      exit(1);
    }
    simhit_py_isLoaded = true;
  }
  return *simhit_py_;
}
const std::vector<std::vector<int> > &Trktree::see_simTrkIdx() {
  if (not see_simTrkIdx_isLoaded) {
    if (see_simTrkIdx_branch != 0) {
      see_simTrkIdx_branch->GetEntry(index);
    } else {
      printf("branch see_simTrkIdx_branch does not exist!\n");
      exit(1);
    }
    see_simTrkIdx_isLoaded = true;
  }
  return *see_simTrkIdx_;
}
const std::vector<float> &Trktree::see_stateTrajGlbZ() {
  if (not see_stateTrajGlbZ_isLoaded) {
    if (see_stateTrajGlbZ_branch != 0) {
      see_stateTrajGlbZ_branch->GetEntry(index);
    } else {
      printf("branch see_stateTrajGlbZ_branch does not exist!\n");
      exit(1);
    }
    see_stateTrajGlbZ_isLoaded = true;
  }
  return *see_stateTrajGlbZ_;
}
const std::vector<float> &Trktree::see_stateTrajGlbX() {
  if (not see_stateTrajGlbX_isLoaded) {
    if (see_stateTrajGlbX_branch != 0) {
      see_stateTrajGlbX_branch->GetEntry(index);
    } else {
      printf("branch see_stateTrajGlbX_branch does not exist!\n");
      exit(1);
    }
    see_stateTrajGlbX_isLoaded = true;
  }
  return *see_stateTrajGlbX_;
}
const std::vector<float> &Trktree::see_stateTrajGlbY() {
  if (not see_stateTrajGlbY_isLoaded) {
    if (see_stateTrajGlbY_branch != 0) {
      see_stateTrajGlbY_branch->GetEntry(index);
    } else {
      printf("branch see_stateTrajGlbY_branch does not exist!\n");
      exit(1);
    }
    see_stateTrajGlbY_isLoaded = true;
  }
  return *see_stateTrajGlbY_;
}
const std::vector<unsigned int> &Trktree::trk_originalAlgo() {
  if (not trk_originalAlgo_isLoaded) {
    if (trk_originalAlgo_branch != 0) {
      trk_originalAlgo_branch->GetEntry(index);
    } else {
      printf("branch trk_originalAlgo_branch does not exist!\n");
      exit(1);
    }
    trk_originalAlgo_isLoaded = true;
  }
  return *trk_originalAlgo_;
}
const std::vector<unsigned int> &Trktree::trk_nPixel() {
  if (not trk_nPixel_isLoaded) {
    if (trk_nPixel_branch != 0) {
      trk_nPixel_branch->GetEntry(index);
    } else {
      printf("branch trk_nPixel_branch does not exist!\n");
      exit(1);
    }
    trk_nPixel_isLoaded = true;
  }
  return *trk_nPixel_;
}
const std::vector<float> &Trktree::see_stateCcov14() {
  if (not see_stateCcov14_isLoaded) {
    if (see_stateCcov14_branch != 0) {
      see_stateCcov14_branch->GetEntry(index);
    } else {
      printf("branch see_stateCcov14_branch does not exist!\n");
      exit(1);
    }
    see_stateCcov14_isLoaded = true;
  }
  return *see_stateCcov14_;
}
const std::vector<float> &Trktree::see_stateCcov15() {
  if (not see_stateCcov15_isLoaded) {
    if (see_stateCcov15_branch != 0) {
      see_stateCcov15_branch->GetEntry(index);
    } else {
      printf("branch see_stateCcov15_branch does not exist!\n");
      exit(1);
    }
    see_stateCcov15_isLoaded = true;
  }
  return *see_stateCcov15_;
}
const std::vector<float> &Trktree::trk_phiErr() {
  if (not trk_phiErr_isLoaded) {
    if (trk_phiErr_branch != 0) {
      trk_phiErr_branch->GetEntry(index);
    } else {
      printf("branch trk_phiErr_branch does not exist!\n");
      exit(1);
    }
    trk_phiErr_isLoaded = true;
  }
  return *trk_phiErr_;
}
const std::vector<float> &Trktree::see_stateCcov13() {
  if (not see_stateCcov13_isLoaded) {
    if (see_stateCcov13_branch != 0) {
      see_stateCcov13_branch->GetEntry(index);
    } else {
      printf("branch see_stateCcov13_branch does not exist!\n");
      exit(1);
    }
    see_stateCcov13_isLoaded = true;
  }
  return *see_stateCcov13_;
}
const std::vector<std::vector<float> > &Trktree::pix_chargeFraction() {
  if (not pix_chargeFraction_isLoaded) {
    if (pix_chargeFraction_branch != 0) {
      pix_chargeFraction_branch->GetEntry(index);
    } else {
      printf("branch pix_chargeFraction_branch does not exist!\n");
      exit(1);
    }
    pix_chargeFraction_isLoaded = true;
  }
  return *pix_chargeFraction_;
}
const std::vector<int> &Trktree::trk_q() {
  if (not trk_q_isLoaded) {
    if (trk_q_branch != 0) {
      trk_q_branch->GetEntry(index);
    } else {
      printf("branch trk_q_branch does not exist!\n");
      exit(1);
    }
    trk_q_isLoaded = true;
  }
  return *trk_q_;
}
const std::vector<std::vector<int> > &Trktree::sim_seedIdx() {
  if (not sim_seedIdx_isLoaded) {
    if (sim_seedIdx_branch != 0) {
      sim_seedIdx_branch->GetEntry(index);
    } else {
      printf("branch sim_seedIdx_branch does not exist!\n");
      exit(1);
    }
    sim_seedIdx_isLoaded = true;
  }
  return *sim_seedIdx_;
}
const std::vector<float> &Trktree::see_dzErr() {
  if (not see_dzErr_isLoaded) {
    if (see_dzErr_branch != 0) {
      see_dzErr_branch->GetEntry(index);
    } else {
      printf("branch see_dzErr_branch does not exist!\n");
      exit(1);
    }
    see_dzErr_isLoaded = true;
  }
  return *see_dzErr_;
}
const std::vector<unsigned int> &Trktree::sim_nRecoClusters() {
  if (not sim_nRecoClusters_isLoaded) {
    if (sim_nRecoClusters_branch != 0) {
      sim_nRecoClusters_branch->GetEntry(index);
    } else {
      printf("branch sim_nRecoClusters_branch does not exist!\n");
      exit(1);
    }
    sim_nRecoClusters_isLoaded = true;
  }
  return *sim_nRecoClusters_;
}
const unsigned int &Trktree::run() {
  if (not run_isLoaded) {
    if (run_branch != 0) {
      run_branch->GetEntry(index);
    } else {
      printf("branch run_branch does not exist!\n");
      exit(1);
    }
    run_isLoaded = true;
  }
  return run_;
}
const std::vector<std::vector<float> > &Trktree::ph2_xySignificance() {
  if (not ph2_xySignificance_isLoaded) {
    if (ph2_xySignificance_branch != 0) {
      ph2_xySignificance_branch->GetEntry(index);
    } else {
      printf("branch ph2_xySignificance_branch does not exist!\n");
      exit(1);
    }
    ph2_xySignificance_isLoaded = true;
  }
  return *ph2_xySignificance_;
}
const std::vector<float> &Trktree::trk_nChi2() {
  if (not trk_nChi2_isLoaded) {
    if (trk_nChi2_branch != 0) {
      trk_nChi2_branch->GetEntry(index);
    } else {
      printf("branch trk_nChi2_branch does not exist!\n");
      exit(1);
    }
    trk_nChi2_isLoaded = true;
  }
  return *trk_nChi2_;
}
const std::vector<unsigned short> &Trktree::pix_layer() {
  if (not pix_layer_isLoaded) {
    if (pix_layer_branch != 0) {
      pix_layer_branch->GetEntry(index);
    } else {
      printf("branch pix_layer_branch does not exist!\n");
      exit(1);
    }
    pix_layer_isLoaded = true;
  }
  return *pix_layer_;
}
const std::vector<std::vector<float> > &Trktree::pix_xySignificance() {
  if (not pix_xySignificance_isLoaded) {
    if (pix_xySignificance_branch != 0) {
      pix_xySignificance_branch->GetEntry(index);
    } else {
      printf("branch pix_xySignificance_branch does not exist!\n");
      exit(1);
    }
    pix_xySignificance_isLoaded = true;
  }
  return *pix_xySignificance_;
}
const std::vector<float> &Trktree::sim_pca_eta() {
  if (not sim_pca_eta_isLoaded) {
    if (sim_pca_eta_branch != 0) {
      sim_pca_eta_branch->GetEntry(index);
    } else {
      printf("branch sim_pca_eta_branch does not exist!\n");
      exit(1);
    }
    sim_pca_eta_isLoaded = true;
  }
  return *sim_pca_eta_;
}
const std::vector<float> &Trktree::see_bestSimTrkShareFrac() {
  if (not see_bestSimTrkShareFrac_isLoaded) {
    if (see_bestSimTrkShareFrac_branch != 0) {
      see_bestSimTrkShareFrac_branch->GetEntry(index);
    } else {
      printf("branch see_bestSimTrkShareFrac_branch does not exist!\n");
      exit(1);
    }
    see_bestSimTrkShareFrac_isLoaded = true;
  }
  return *see_bestSimTrkShareFrac_;
}
const std::vector<float> &Trktree::see_etaErr() {
  if (not see_etaErr_isLoaded) {
    if (see_etaErr_branch != 0) {
      see_etaErr_branch->GetEntry(index);
    } else {
      printf("branch see_etaErr_branch does not exist!\n");
      exit(1);
    }
    see_etaErr_isLoaded = true;
  }
  return *see_etaErr_;
}
const std::vector<float> &Trktree::trk_bestSimTrkShareFracSimDenom() {
  if (not trk_bestSimTrkShareFracSimDenom_isLoaded) {
    if (trk_bestSimTrkShareFracSimDenom_branch != 0) {
      trk_bestSimTrkShareFracSimDenom_branch->GetEntry(index);
    } else {
      printf("branch trk_bestSimTrkShareFracSimDenom_branch does not exist!\n");
      exit(1);
    }
    trk_bestSimTrkShareFracSimDenom_isLoaded = true;
  }
  return *trk_bestSimTrkShareFracSimDenom_;
}
const float &Trktree::bsp_sigmaz() {
  if (not bsp_sigmaz_isLoaded) {
    if (bsp_sigmaz_branch != 0) {
      bsp_sigmaz_branch->GetEntry(index);
    } else {
      printf("branch bsp_sigmaz_branch does not exist!\n");
      exit(1);
    }
    bsp_sigmaz_isLoaded = true;
  }
  return bsp_sigmaz_;
}
const float &Trktree::bsp_sigmay() {
  if (not bsp_sigmay_isLoaded) {
    if (bsp_sigmay_branch != 0) {
      bsp_sigmay_branch->GetEntry(index);
    } else {
      printf("branch bsp_sigmay_branch does not exist!\n");
      exit(1);
    }
    bsp_sigmay_isLoaded = true;
  }
  return bsp_sigmay_;
}
const float &Trktree::bsp_sigmax() {
  if (not bsp_sigmax_isLoaded) {
    if (bsp_sigmax_branch != 0) {
      bsp_sigmax_branch->GetEntry(index);
    } else {
      printf("branch bsp_sigmax_branch does not exist!\n");
      exit(1);
    }
    bsp_sigmax_isLoaded = true;
  }
  return bsp_sigmax_;
}
const std::vector<unsigned short> &Trktree::pix_ladder() {
  if (not pix_ladder_isLoaded) {
    if (pix_ladder_branch != 0) {
      pix_ladder_branch->GetEntry(index);
    } else {
      printf("branch pix_ladder_branch does not exist!\n");
      exit(1);
    }
    pix_ladder_isLoaded = true;
  }
  return *pix_ladder_;
}
const std::vector<unsigned short> &Trktree::trk_qualityMask() {
  if (not trk_qualityMask_isLoaded) {
    if (trk_qualityMask_branch != 0) {
      trk_qualityMask_branch->GetEntry(index);
    } else {
      printf("branch trk_qualityMask_branch does not exist!\n");
      exit(1);
    }
    trk_qualityMask_isLoaded = true;
  }
  return *trk_qualityMask_;
}
const std::vector<float> &Trktree::trk_ndof() {
  if (not trk_ndof_isLoaded) {
    if (trk_ndof_branch != 0) {
      trk_ndof_branch->GetEntry(index);
    } else {
      printf("branch trk_ndof_branch does not exist!\n");
      exit(1);
    }
    trk_ndof_isLoaded = true;
  }
  return *trk_ndof_;
}
const std::vector<unsigned short> &Trktree::pix_subdet() {
  if (not pix_subdet_isLoaded) {
    if (pix_subdet_branch != 0) {
      pix_subdet_branch->GetEntry(index);
    } else {
      printf("branch pix_subdet_branch does not exist!\n");
      exit(1);
    }
    pix_subdet_isLoaded = true;
  }
  return *pix_subdet_;
}
const std::vector<std::vector<int> > &Trktree::ph2_seeIdx() {
  if (not ph2_seeIdx_isLoaded) {
    if (ph2_seeIdx_branch != 0) {
      ph2_seeIdx_branch->GetEntry(index);
    } else {
      printf("branch ph2_seeIdx_branch does not exist!\n");
      exit(1);
    }
    ph2_seeIdx_isLoaded = true;
  }
  return *ph2_seeIdx_;
}
const std::vector<unsigned short> &Trktree::inv_isUpper() {
  if (not inv_isUpper_isLoaded) {
    if (inv_isUpper_branch != 0) {
      inv_isUpper_branch->GetEntry(index);
    } else {
      printf("branch inv_isUpper_branch does not exist!\n");
      exit(1);
    }
    inv_isUpper_isLoaded = true;
  }
  return *inv_isUpper_;
}
const std::vector<float> &Trktree::ph2_zx() {
  if (not ph2_zx_isLoaded) {
    if (ph2_zx_branch != 0) {
      ph2_zx_branch->GetEntry(index);
    } else {
      printf("branch ph2_zx_branch does not exist!\n");
      exit(1);
    }
    ph2_zx_isLoaded = true;
  }
  return *ph2_zx_;
}
const std::vector<std::vector<int> > &Trktree::pix_trkIdx() {
  if (not pix_trkIdx_isLoaded) {
    if (pix_trkIdx_branch != 0) {
      pix_trkIdx_branch->GetEntry(index);
    } else {
      printf("branch pix_trkIdx_branch does not exist!\n");
      exit(1);
    }
    pix_trkIdx_isLoaded = true;
  }
  return *pix_trkIdx_;
}
const std::vector<unsigned int> &Trktree::trk_nOuterLost() {
  if (not trk_nOuterLost_isLoaded) {
    if (trk_nOuterLost_branch != 0) {
      trk_nOuterLost_branch->GetEntry(index);
    } else {
      printf("branch trk_nOuterLost_branch does not exist!\n");
      exit(1);
    }
    trk_nOuterLost_isLoaded = true;
  }
  return *trk_nOuterLost_;
}
const std::vector<unsigned short> &Trktree::inv_panel() {
  if (not inv_panel_isLoaded) {
    if (inv_panel_branch != 0) {
      inv_panel_branch->GetEntry(index);
    } else {
      printf("branch inv_panel_branch does not exist!\n");
      exit(1);
    }
    inv_panel_isLoaded = true;
  }
  return *inv_panel_;
}
const std::vector<float> &Trktree::vtx_z() {
  if (not vtx_z_isLoaded) {
    if (vtx_z_branch != 0) {
      vtx_z_branch->GetEntry(index);
    } else {
      printf("branch vtx_z_branch does not exist!\n");
      exit(1);
    }
    vtx_z_isLoaded = true;
  }
  return *vtx_z_;
}
const std::vector<unsigned short> &Trktree::simhit_layer() {
  if (not simhit_layer_isLoaded) {
    if (simhit_layer_branch != 0) {
      simhit_layer_branch->GetEntry(index);
    } else {
      printf("branch simhit_layer_branch does not exist!\n");
      exit(1);
    }
    simhit_layer_isLoaded = true;
  }
  return *simhit_layer_;
}
const std::vector<float> &Trktree::vtx_y() {
  if (not vtx_y_isLoaded) {
    if (vtx_y_branch != 0) {
      vtx_y_branch->GetEntry(index);
    } else {
      printf("branch vtx_y_branch does not exist!\n");
      exit(1);
    }
    vtx_y_isLoaded = true;
  }
  return *vtx_y_;
}
const std::vector<short> &Trktree::ph2_isBarrel() {
  if (not ph2_isBarrel_isLoaded) {
    if (ph2_isBarrel_branch != 0) {
      ph2_isBarrel_branch->GetEntry(index);
    } else {
      printf("branch ph2_isBarrel_branch does not exist!\n");
      exit(1);
    }
    ph2_isBarrel_isLoaded = true;
  }
  return *ph2_isBarrel_;
}
const std::vector<std::vector<int> > &Trktree::pix_seeIdx() {
  if (not pix_seeIdx_isLoaded) {
    if (pix_seeIdx_branch != 0) {
      pix_seeIdx_branch->GetEntry(index);
    } else {
      printf("branch pix_seeIdx_branch does not exist!\n");
      exit(1);
    }
    pix_seeIdx_isLoaded = true;
  }
  return *pix_seeIdx_;
}
const std::vector<int> &Trktree::trk_bestFromFirstHitSimTrkIdx() {
  if (not trk_bestFromFirstHitSimTrkIdx_isLoaded) {
    if (trk_bestFromFirstHitSimTrkIdx_branch != 0) {
      trk_bestFromFirstHitSimTrkIdx_branch->GetEntry(index);
    } else {
      printf("branch trk_bestFromFirstHitSimTrkIdx_branch does not exist!\n");
      exit(1);
    }
    trk_bestFromFirstHitSimTrkIdx_isLoaded = true;
  }
  return *trk_bestFromFirstHitSimTrkIdx_;
}
const std::vector<float> &Trktree::simhit_px() {
  if (not simhit_px_isLoaded) {
    if (simhit_px_branch != 0) {
      simhit_px_branch->GetEntry(index);
    } else {
      printf("branch simhit_px_branch does not exist!\n");
      exit(1);
    }
    simhit_px_isLoaded = true;
  }
  return *simhit_px_;
}
const std::vector<float> &Trktree::see_stateTrajX() {
  if (not see_stateTrajX_isLoaded) {
    if (see_stateTrajX_branch != 0) {
      see_stateTrajX_branch->GetEntry(index);
    } else {
      printf("branch see_stateTrajX_branch does not exist!\n");
      exit(1);
    }
    see_stateTrajX_isLoaded = true;
  }
  return *see_stateTrajX_;
}
const std::vector<float> &Trktree::see_stateTrajY() {
  if (not see_stateTrajY_isLoaded) {
    if (see_stateTrajY_branch != 0) {
      see_stateTrajY_branch->GetEntry(index);
    } else {
      printf("branch see_stateTrajY_branch does not exist!\n");
      exit(1);
    }
    see_stateTrajY_isLoaded = true;
  }
  return *see_stateTrajY_;
}
const std::vector<unsigned int> &Trktree::trk_nOuterInactive() {
  if (not trk_nOuterInactive_isLoaded) {
    if (trk_nOuterInactive_branch != 0) {
      trk_nOuterInactive_branch->GetEntry(index);
    } else {
      printf("branch trk_nOuterInactive_branch does not exist!\n");
      exit(1);
    }
    trk_nOuterInactive_isLoaded = true;
  }
  return *trk_nOuterInactive_;
}
const std::vector<float> &Trktree::sim_pca_dxy() {
  if (not sim_pca_dxy_isLoaded) {
    if (sim_pca_dxy_branch != 0) {
      sim_pca_dxy_branch->GetEntry(index);
    } else {
      printf("branch sim_pca_dxy_branch does not exist!\n");
      exit(1);
    }
    sim_pca_dxy_isLoaded = true;
  }
  return *sim_pca_dxy_;
}
const std::vector<unsigned int> &Trktree::trk_algo() {
  if (not trk_algo_isLoaded) {
    if (trk_algo_branch != 0) {
      trk_algo_branch->GetEntry(index);
    } else {
      printf("branch trk_algo_branch does not exist!\n");
      exit(1);
    }
    trk_algo_isLoaded = true;
  }
  return *trk_algo_;
}
const std::vector<std::vector<int> > &Trktree::trk_hitType() {
  if (not trk_hitType_isLoaded) {
    if (trk_hitType_branch != 0) {
      trk_hitType_branch->GetEntry(index);
    } else {
      printf("branch trk_hitType_branch does not exist!\n");
      exit(1);
    }
    trk_hitType_isLoaded = true;
  }
  return *trk_hitType_;
}
const std::vector<float> &Trktree::trk_bestFromFirstHitSimTrkShareFrac() {
  if (not trk_bestFromFirstHitSimTrkShareFrac_isLoaded) {
    if (trk_bestFromFirstHitSimTrkShareFrac_branch != 0) {
      trk_bestFromFirstHitSimTrkShareFrac_branch->GetEntry(index);
    } else {
      printf("branch trk_bestFromFirstHitSimTrkShareFrac_branch does not exist!\n");
      exit(1);
    }
    trk_bestFromFirstHitSimTrkShareFrac_isLoaded = true;
  }
  return *trk_bestFromFirstHitSimTrkShareFrac_;
}
const std::vector<short> &Trktree::inv_isBarrel() {
  if (not inv_isBarrel_isLoaded) {
    if (inv_isBarrel_branch != 0) {
      inv_isBarrel_branch->GetEntry(index);
    } else {
      printf("branch inv_isBarrel_branch does not exist!\n");
      exit(1);
    }
    inv_isBarrel_isLoaded = true;
  }
  return *inv_isBarrel_;
}
const std::vector<int> &Trktree::simvtx_event() {
  if (not simvtx_event_isLoaded) {
    if (simvtx_event_branch != 0) {
      simvtx_event_branch->GetEntry(index);
    } else {
      printf("branch simvtx_event_branch does not exist!\n");
      exit(1);
    }
    simvtx_event_isLoaded = true;
  }
  return *simvtx_event_;
}
const std::vector<float> &Trktree::ph2_z() {
  if (not ph2_z_isLoaded) {
    if (ph2_z_branch != 0) {
      ph2_z_branch->GetEntry(index);
    } else {
      printf("branch ph2_z_branch does not exist!\n");
      exit(1);
    }
    ph2_z_isLoaded = true;
  }
  return *ph2_z_;
}
const std::vector<float> &Trktree::ph2_x() {
  if (not ph2_x_isLoaded) {
    if (ph2_x_branch != 0) {
      ph2_x_branch->GetEntry(index);
    } else {
      printf("branch ph2_x_branch does not exist!\n");
      exit(1);
    }
    ph2_x_isLoaded = true;
  }
  return *ph2_x_;
}
const std::vector<float> &Trktree::ph2_y() {
  if (not ph2_y_isLoaded) {
    if (ph2_y_branch != 0) {
      ph2_y_branch->GetEntry(index);
    } else {
      printf("branch ph2_y_branch does not exist!\n");
      exit(1);
    }
    ph2_y_isLoaded = true;
  }
  return *ph2_y_;
}
const std::vector<std::vector<int> > &Trktree::sim_genPdgIds() {
  if (not sim_genPdgIds_isLoaded) {
    if (sim_genPdgIds_branch != 0) {
      sim_genPdgIds_branch->GetEntry(index);
    } else {
      printf("branch sim_genPdgIds_branch does not exist!\n");
      exit(1);
    }
    sim_genPdgIds_isLoaded = true;
  }
  return *sim_genPdgIds_;
}
const std::vector<float> &Trktree::trk_mva() {
  if (not trk_mva_isLoaded) {
    if (trk_mva_branch != 0) {
      trk_mva_branch->GetEntry(index);
    } else {
      printf("branch trk_mva_branch does not exist!\n");
      exit(1);
    }
    trk_mva_isLoaded = true;
  }
  return *trk_mva_;
}
const std::vector<float> &Trktree::see_stateCcov24() {
  if (not see_stateCcov24_isLoaded) {
    if (see_stateCcov24_branch != 0) {
      see_stateCcov24_branch->GetEntry(index);
    } else {
      printf("branch see_stateCcov24_branch does not exist!\n");
      exit(1);
    }
    see_stateCcov24_isLoaded = true;
  }
  return *see_stateCcov24_;
}
const std::vector<float> &Trktree::trk_dzClosestPV() {
  if (not trk_dzClosestPV_isLoaded) {
    if (trk_dzClosestPV_branch != 0) {
      trk_dzClosestPV_branch->GetEntry(index);
    } else {
      printf("branch trk_dzClosestPV_branch does not exist!\n");
      exit(1);
    }
    trk_dzClosestPV_isLoaded = true;
  }
  return *trk_dzClosestPV_;
}
const std::vector<unsigned int> &Trktree::see_nCluster() {
  if (not see_nCluster_isLoaded) {
    if (see_nCluster_branch != 0) {
      see_nCluster_branch->GetEntry(index);
    } else {
      printf("branch see_nCluster_branch does not exist!\n");
      exit(1);
    }
    see_nCluster_isLoaded = true;
  }
  return *see_nCluster_;
}
const std::vector<unsigned short> &Trktree::inv_rod() {
  if (not inv_rod_isLoaded) {
    if (inv_rod_branch != 0) {
      inv_rod_branch->GetEntry(index);
    } else {
      printf("branch inv_rod_branch does not exist!\n");
      exit(1);
    }
    inv_rod_isLoaded = true;
  }
  return *inv_rod_;
}
const std::vector<std::vector<int> > &Trktree::trk_hitIdx() {
  if (not trk_hitIdx_isLoaded) {
    if (trk_hitIdx_branch != 0) {
      trk_hitIdx_branch->GetEntry(index);
    } else {
      printf("branch trk_hitIdx_branch does not exist!\n");
      exit(1);
    }
    trk_hitIdx_isLoaded = true;
  }
  return *trk_hitIdx_;
}
const std::vector<float> &Trktree::see_stateCcov22() {
  if (not see_stateCcov22_isLoaded) {
    if (see_stateCcov22_branch != 0) {
      see_stateCcov22_branch->GetEntry(index);
    } else {
      printf("branch see_stateCcov22_branch does not exist!\n");
      exit(1);
    }
    see_stateCcov22_isLoaded = true;
  }
  return *see_stateCcov22_;
}
const std::vector<unsigned short> &Trktree::pix_simType() {
  if (not pix_simType_isLoaded) {
    if (pix_simType_branch != 0) {
      pix_simType_branch->GetEntry(index);
    } else {
      printf("branch pix_simType_branch does not exist!\n");
      exit(1);
    }
    pix_simType_isLoaded = true;
  }
  return *pix_simType_;
}
const std::vector<unsigned short> &Trktree::simhit_ring() {
  if (not simhit_ring_isLoaded) {
    if (simhit_ring_branch != 0) {
      simhit_ring_branch->GetEntry(index);
    } else {
      printf("branch simhit_ring_branch does not exist!\n");
      exit(1);
    }
    simhit_ring_isLoaded = true;
  }
  return *simhit_ring_;
}
const std::vector<float> &Trktree::trk_outer_px() {
  if (not trk_outer_px_isLoaded) {
    if (trk_outer_px_branch != 0) {
      trk_outer_px_branch->GetEntry(index);
    } else {
      printf("branch trk_outer_px_branch does not exist!\n");
      exit(1);
    }
    trk_outer_px_isLoaded = true;
  }
  return *trk_outer_px_;
}
const std::vector<float> &Trktree::trk_outer_py() {
  if (not trk_outer_py_isLoaded) {
    if (trk_outer_py_branch != 0) {
      trk_outer_py_branch->GetEntry(index);
    } else {
      printf("branch trk_outer_py_branch does not exist!\n");
      exit(1);
    }
    trk_outer_py_isLoaded = true;
  }
  return *trk_outer_py_;
}
const std::vector<float> &Trktree::trk_outer_pz() {
  if (not trk_outer_pz_isLoaded) {
    if (trk_outer_pz_branch != 0) {
      trk_outer_pz_branch->GetEntry(index);
    } else {
      printf("branch trk_outer_pz_branch does not exist!\n");
      exit(1);
    }
    trk_outer_pz_isLoaded = true;
  }
  return *trk_outer_pz_;
}
const std::vector<float> &Trktree::ph2_zz() {
  if (not ph2_zz_isLoaded) {
    if (ph2_zz_branch != 0) {
      ph2_zz_branch->GetEntry(index);
    } else {
      printf("branch ph2_zz_branch does not exist!\n");
      exit(1);
    }
    ph2_zz_isLoaded = true;
  }
  return *ph2_zz_;
}
const std::vector<float> &Trktree::trk_outer_pt() {
  if (not trk_outer_pt_isLoaded) {
    if (trk_outer_pt_branch != 0) {
      trk_outer_pt_branch->GetEntry(index);
    } else {
      printf("branch trk_outer_pt_branch does not exist!\n");
      exit(1);
    }
    trk_outer_pt_isLoaded = true;
  }
  return *trk_outer_pt_;
}
const std::vector<unsigned int> &Trktree::trk_n3DLay() {
  if (not trk_n3DLay_isLoaded) {
    if (trk_n3DLay_branch != 0) {
      trk_n3DLay_branch->GetEntry(index);
    } else {
      printf("branch trk_n3DLay_branch does not exist!\n");
      exit(1);
    }
    trk_n3DLay_isLoaded = true;
  }
  return *trk_n3DLay_;
}
const std::vector<unsigned int> &Trktree::trk_nValid() {
  if (not trk_nValid_isLoaded) {
    if (trk_nValid_branch != 0) {
      trk_nValid_branch->GetEntry(index);
    } else {
      printf("branch trk_nValid_branch does not exist!\n");
      exit(1);
    }
    trk_nValid_isLoaded = true;
  }
  return *trk_nValid_;
}
const std::vector<float> &Trktree::see_ptErr() {
  if (not see_ptErr_isLoaded) {
    if (see_ptErr_branch != 0) {
      see_ptErr_branch->GetEntry(index);
    } else {
      printf("branch see_ptErr_branch does not exist!\n");
      exit(1);
    }
    see_ptErr_isLoaded = true;
  }
  return *see_ptErr_;
}
const std::vector<float> &Trktree::see_stateTrajGlbPx() {
  if (not see_stateTrajGlbPx_isLoaded) {
    if (see_stateTrajGlbPx_branch != 0) {
      see_stateTrajGlbPx_branch->GetEntry(index);
    } else {
      printf("branch see_stateTrajGlbPx_branch does not exist!\n");
      exit(1);
    }
    see_stateTrajGlbPx_isLoaded = true;
  }
  return *see_stateTrajGlbPx_;
}
const std::vector<unsigned short> &Trktree::ph2_simType() {
  if (not ph2_simType_isLoaded) {
    if (ph2_simType_branch != 0) {
      ph2_simType_branch->GetEntry(index);
    } else {
      printf("branch ph2_simType_branch does not exist!\n");
      exit(1);
    }
    ph2_simType_isLoaded = true;
  }
  return *ph2_simType_;
}
const std::vector<float> &Trktree::trk_bestFromFirstHitSimTrkShareFracSimClusterDenom() {
  if (not trk_bestFromFirstHitSimTrkShareFracSimClusterDenom_isLoaded) {
    if (trk_bestFromFirstHitSimTrkShareFracSimClusterDenom_branch != 0) {
      trk_bestFromFirstHitSimTrkShareFracSimClusterDenom_branch->GetEntry(index);
    } else {
      printf("branch trk_bestFromFirstHitSimTrkShareFracSimClusterDenom_branch does not exist!\n");
      exit(1);
    }
    trk_bestFromFirstHitSimTrkShareFracSimClusterDenom_isLoaded = true;
  }
  return *trk_bestFromFirstHitSimTrkShareFracSimClusterDenom_;
}
const std::vector<float> &Trktree::sim_hits() {
  if (not sim_hits_isLoaded) {
    if (sim_hits_branch != 0) {
      sim_hits_branch->GetEntry(index);
    } else {
      printf("branch sim_hits_branch does not exist!\n");
      exit(1);
    }
    sim_hits_isLoaded = true;
  }
  return *sim_hits_;
}
const std::vector<float> &Trktree::sim_len() {
  if (not sim_len_isLoaded) {
    if (sim_len_branch != 0) {
      sim_len_branch->GetEntry(index);
    } else {
      printf("branch sim_len_branch does not exist!\n");
      exit(1);
    }
    sim_len_isLoaded = true;
  }
  return *sim_len_;
}
const std::vector<float> &Trktree::sim_lengap() {
  if (not sim_lengap_isLoaded) {
    if (sim_lengap_branch != 0) {
      sim_lengap_branch->GetEntry(index);
    } else {
      printf("branch sim_lengap_branch does not exist!\n");
      exit(1);
    }
    sim_lengap_isLoaded = true;
  }
  return *sim_lengap_;
}
const std::vector<float> &Trktree::simvtx_x() {
  if (not simvtx_x_isLoaded) {
    if (simvtx_x_branch != 0) {
      simvtx_x_branch->GetEntry(index);
    } else {
      printf("branch simvtx_x_branch does not exist!\n");
      exit(1);
    }
    simvtx_x_isLoaded = true;
  }
  return *simvtx_x_;
}
const std::vector<float> &Trktree::trk_pz() {
  if (not trk_pz_isLoaded) {
    if (trk_pz_branch != 0) {
      trk_pz_branch->GetEntry(index);
    } else {
      printf("branch trk_pz_branch does not exist!\n");
      exit(1);
    }
    trk_pz_isLoaded = true;
  }
  return *trk_pz_;
}
const std::vector<float> &Trktree::see_bestFromFirstHitSimTrkShareFrac() {
  if (not see_bestFromFirstHitSimTrkShareFrac_isLoaded) {
    if (see_bestFromFirstHitSimTrkShareFrac_branch != 0) {
      see_bestFromFirstHitSimTrkShareFrac_branch->GetEntry(index);
    } else {
      printf("branch see_bestFromFirstHitSimTrkShareFrac_branch does not exist!\n");
      exit(1);
    }
    see_bestFromFirstHitSimTrkShareFrac_isLoaded = true;
  }
  return *see_bestFromFirstHitSimTrkShareFrac_;
}
const std::vector<float> &Trktree::trk_px() {
  if (not trk_px_isLoaded) {
    if (trk_px_branch != 0) {
      trk_px_branch->GetEntry(index);
    } else {
      printf("branch trk_px_branch does not exist!\n");
      exit(1);
    }
    trk_px_isLoaded = true;
  }
  return *trk_px_;
}
const std::vector<float> &Trktree::trk_py() {
  if (not trk_py_isLoaded) {
    if (trk_py_branch != 0) {
      trk_py_branch->GetEntry(index);
    } else {
      printf("branch trk_py_branch does not exist!\n");
      exit(1);
    }
    trk_py_isLoaded = true;
  }
  return *trk_py_;
}
const std::vector<int> &Trktree::trk_vtxIdx() {
  if (not trk_vtxIdx_isLoaded) {
    if (trk_vtxIdx_branch != 0) {
      trk_vtxIdx_branch->GetEntry(index);
    } else {
      printf("branch trk_vtxIdx_branch does not exist!\n");
      exit(1);
    }
    trk_vtxIdx_isLoaded = true;
  }
  return *trk_vtxIdx_;
}
const std::vector<unsigned int> &Trktree::sim_nPixel() {
  if (not sim_nPixel_isLoaded) {
    if (sim_nPixel_branch != 0) {
      sim_nPixel_branch->GetEntry(index);
    } else {
      printf("branch sim_nPixel_branch does not exist!\n");
      exit(1);
    }
    sim_nPixel_isLoaded = true;
  }
  return *sim_nPixel_;
}
const std::vector<float> &Trktree::vtx_chi2() {
  if (not vtx_chi2_isLoaded) {
    if (vtx_chi2_branch != 0) {
      vtx_chi2_branch->GetEntry(index);
    } else {
      printf("branch vtx_chi2_branch does not exist!\n");
      exit(1);
    }
    vtx_chi2_isLoaded = true;
  }
  return *vtx_chi2_;
}
const std::vector<unsigned short> &Trktree::ph2_ring() {
  if (not ph2_ring_isLoaded) {
    if (ph2_ring_branch != 0) {
      ph2_ring_branch->GetEntry(index);
    } else {
      printf("branch ph2_ring_branch does not exist!\n");
      exit(1);
    }
    ph2_ring_isLoaded = true;
  }
  return *ph2_ring_;
}
const std::vector<float> &Trktree::trk_pt() {
  if (not trk_pt_isLoaded) {
    if (trk_pt_branch != 0) {
      trk_pt_branch->GetEntry(index);
    } else {
      printf("branch trk_pt_branch does not exist!\n");
      exit(1);
    }
    trk_pt_isLoaded = true;
  }
  return *trk_pt_;
}
const std::vector<float> &Trktree::see_stateCcov44() {
  if (not see_stateCcov44_isLoaded) {
    if (see_stateCcov44_branch != 0) {
      see_stateCcov44_branch->GetEntry(index);
    } else {
      printf("branch see_stateCcov44_branch does not exist!\n");
      exit(1);
    }
    see_stateCcov44_isLoaded = true;
  }
  return *see_stateCcov44_;
}
const std::vector<float> &Trktree::ph2_radL() {
  if (not ph2_radL_isLoaded) {
    if (ph2_radL_branch != 0) {
      ph2_radL_branch->GetEntry(index);
    } else {
      printf("branch ph2_radL_branch does not exist!\n");
      exit(1);
    }
    ph2_radL_isLoaded = true;
  }
  return *ph2_radL_;
}
const std::vector<float> &Trktree::vtx_zErr() {
  if (not vtx_zErr_isLoaded) {
    if (vtx_zErr_branch != 0) {
      vtx_zErr_branch->GetEntry(index);
    } else {
      printf("branch vtx_zErr_branch does not exist!\n");
      exit(1);
    }
    vtx_zErr_isLoaded = true;
  }
  return *vtx_zErr_;
}
const std::vector<float> &Trktree::see_px() {
  if (not see_px_isLoaded) {
    if (see_px_branch != 0) {
      see_px_branch->GetEntry(index);
    } else {
      printf("branch see_px_branch does not exist!\n");
      exit(1);
    }
    see_px_isLoaded = true;
  }
  return *see_px_;
}
const std::vector<float> &Trktree::see_pz() {
  if (not see_pz_isLoaded) {
    if (see_pz_branch != 0) {
      see_pz_branch->GetEntry(index);
    } else {
      printf("branch see_pz_branch does not exist!\n");
      exit(1);
    }
    see_pz_isLoaded = true;
  }
  return *see_pz_;
}
const std::vector<float> &Trktree::see_eta() {
  if (not see_eta_isLoaded) {
    if (see_eta_branch != 0) {
      see_eta_branch->GetEntry(index);
    } else {
      printf("branch see_eta_branch does not exist!\n");
      exit(1);
    }
    see_eta_isLoaded = true;
  }
  return *see_eta_;
}
const std::vector<int> &Trktree::simvtx_bunchCrossing() {
  if (not simvtx_bunchCrossing_isLoaded) {
    if (simvtx_bunchCrossing_branch != 0) {
      simvtx_bunchCrossing_branch->GetEntry(index);
    } else {
      printf("branch simvtx_bunchCrossing_branch does not exist!\n");
      exit(1);
    }
    simvtx_bunchCrossing_isLoaded = true;
  }
  return *simvtx_bunchCrossing_;
}
const std::vector<float> &Trktree::sim_pca_dz() {
  if (not sim_pca_dz_isLoaded) {
    if (sim_pca_dz_branch != 0) {
      sim_pca_dz_branch->GetEntry(index);
    } else {
      printf("branch sim_pca_dz_branch does not exist!\n");
      exit(1);
    }
    sim_pca_dz_isLoaded = true;
  }
  return *sim_pca_dz_;
}
const std::vector<float> &Trktree::simvtx_y() {
  if (not simvtx_y_isLoaded) {
    if (simvtx_y_branch != 0) {
      simvtx_y_branch->GetEntry(index);
    } else {
      printf("branch simvtx_y_branch does not exist!\n");
      exit(1);
    }
    simvtx_y_isLoaded = true;
  }
  return *simvtx_y_;
}
const std::vector<unsigned short> &Trktree::inv_isStack() {
  if (not inv_isStack_isLoaded) {
    if (inv_isStack_branch != 0) {
      inv_isStack_branch->GetEntry(index);
    } else {
      printf("branch inv_isStack_branch does not exist!\n");
      exit(1);
    }
    inv_isStack_isLoaded = true;
  }
  return *inv_isStack_;
}
const std::vector<unsigned int> &Trktree::trk_nStrip() {
  if (not trk_nStrip_isLoaded) {
    if (trk_nStrip_branch != 0) {
      trk_nStrip_branch->GetEntry(index);
    } else {
      printf("branch trk_nStrip_branch does not exist!\n");
      exit(1);
    }
    trk_nStrip_isLoaded = true;
  }
  return *trk_nStrip_;
}
const std::vector<float> &Trktree::trk_etaErr() {
  if (not trk_etaErr_isLoaded) {
    if (trk_etaErr_branch != 0) {
      trk_etaErr_branch->GetEntry(index);
    } else {
      printf("branch trk_etaErr_branch does not exist!\n");
      exit(1);
    }
    trk_etaErr_isLoaded = true;
  }
  return *trk_etaErr_;
}
const std::vector<std::vector<float> > &Trktree::trk_simTrkNChi2() {
  if (not trk_simTrkNChi2_isLoaded) {
    if (trk_simTrkNChi2_branch != 0) {
      trk_simTrkNChi2_branch->GetEntry(index);
    } else {
      printf("branch trk_simTrkNChi2_branch does not exist!\n");
      exit(1);
    }
    trk_simTrkNChi2_isLoaded = true;
  }
  return *trk_simTrkNChi2_;
}
const std::vector<float> &Trktree::pix_zz() {
  if (not pix_zz_isLoaded) {
    if (pix_zz_branch != 0) {
      pix_zz_branch->GetEntry(index);
    } else {
      printf("branch pix_zz_branch does not exist!\n");
      exit(1);
    }
    pix_zz_isLoaded = true;
  }
  return *pix_zz_;
}
const std::vector<int> &Trktree::simhit_particle() {
  if (not simhit_particle_isLoaded) {
    if (simhit_particle_branch != 0) {
      simhit_particle_branch->GetEntry(index);
    } else {
      printf("branch simhit_particle_branch does not exist!\n");
      exit(1);
    }
    simhit_particle_isLoaded = true;
  }
  return *simhit_particle_;
}
const std::vector<float> &Trktree::see_dz() {
  if (not see_dz_isLoaded) {
    if (see_dz_branch != 0) {
      see_dz_branch->GetEntry(index);
    } else {
      printf("branch see_dz_branch does not exist!\n");
      exit(1);
    }
    see_dz_isLoaded = true;
  }
  return *see_dz_;
}
const std::vector<float> &Trktree::see_stateTrajPz() {
  if (not see_stateTrajPz_isLoaded) {
    if (see_stateTrajPz_branch != 0) {
      see_stateTrajPz_branch->GetEntry(index);
    } else {
      printf("branch see_stateTrajPz_branch does not exist!\n");
      exit(1);
    }
    see_stateTrajPz_isLoaded = true;
  }
  return *see_stateTrajPz_;
}
const std::vector<float> &Trktree::trk_bestSimTrkShareFrac() {
  if (not trk_bestSimTrkShareFrac_isLoaded) {
    if (trk_bestSimTrkShareFrac_branch != 0) {
      trk_bestSimTrkShareFrac_branch->GetEntry(index);
    } else {
      printf("branch trk_bestSimTrkShareFrac_branch does not exist!\n");
      exit(1);
    }
    trk_bestSimTrkShareFrac_isLoaded = true;
  }
  return *trk_bestSimTrkShareFrac_;
}
const std::vector<float> &Trktree::trk_lambdaErr() {
  if (not trk_lambdaErr_isLoaded) {
    if (trk_lambdaErr_branch != 0) {
      trk_lambdaErr_branch->GetEntry(index);
    } else {
      printf("branch trk_lambdaErr_branch does not exist!\n");
      exit(1);
    }
    trk_lambdaErr_isLoaded = true;
  }
  return *trk_lambdaErr_;
}
const std::vector<std::vector<float> > &Trktree::see_simTrkShareFrac() {
  if (not see_simTrkShareFrac_isLoaded) {
    if (see_simTrkShareFrac_branch != 0) {
      see_simTrkShareFrac_branch->GetEntry(index);
    } else {
      printf("branch see_simTrkShareFrac_branch does not exist!\n");
      exit(1);
    }
    see_simTrkShareFrac_isLoaded = true;
  }
  return *see_simTrkShareFrac_;
}
const std::vector<std::vector<int> > &Trktree::pix_simHitIdx() {
  if (not pix_simHitIdx_isLoaded) {
    if (pix_simHitIdx_branch != 0) {
      pix_simHitIdx_branch->GetEntry(index);
    } else {
      printf("branch pix_simHitIdx_branch does not exist!\n");
      exit(1);
    }
    pix_simHitIdx_isLoaded = true;
  }
  return *pix_simHitIdx_;
}
const std::vector<std::vector<int> > &Trktree::vtx_trkIdx() {
  if (not vtx_trkIdx_isLoaded) {
    if (vtx_trkIdx_branch != 0) {
      vtx_trkIdx_branch->GetEntry(index);
    } else {
      printf("branch vtx_trkIdx_branch does not exist!\n");
      exit(1);
    }
    vtx_trkIdx_isLoaded = true;
  }
  return *vtx_trkIdx_;
}
const std::vector<unsigned short> &Trktree::ph2_rod() {
  if (not ph2_rod_isLoaded) {
    if (ph2_rod_branch != 0) {
      ph2_rod_branch->GetEntry(index);
    } else {
      printf("branch ph2_rod_branch does not exist!\n");
      exit(1);
    }
    ph2_rod_isLoaded = true;
  }
  return *ph2_rod_;
}
const std::vector<float> &Trktree::vtx_ndof() {
  if (not vtx_ndof_isLoaded) {
    if (vtx_ndof_branch != 0) {
      vtx_ndof_branch->GetEntry(index);
    } else {
      printf("branch vtx_ndof_branch does not exist!\n");
      exit(1);
    }
    vtx_ndof_isLoaded = true;
  }
  return *vtx_ndof_;
}
const std::vector<unsigned int> &Trktree::see_nPixel() {
  if (not see_nPixel_isLoaded) {
    if (see_nPixel_branch != 0) {
      see_nPixel_branch->GetEntry(index);
    } else {
      printf("branch see_nPixel_branch does not exist!\n");
      exit(1);
    }
    see_nPixel_isLoaded = true;
  }
  return *see_nPixel_;
}
const std::vector<unsigned int> &Trktree::sim_nStrip() {
  if (not sim_nStrip_isLoaded) {
    if (sim_nStrip_branch != 0) {
      sim_nStrip_branch->GetEntry(index);
    } else {
      printf("branch sim_nStrip_branch does not exist!\n");
      exit(1);
    }
    sim_nStrip_isLoaded = true;
  }
  return *sim_nStrip_;
}
const std::vector<int> &Trktree::sim_bunchCrossing() {
  if (not sim_bunchCrossing_isLoaded) {
    if (sim_bunchCrossing_branch != 0) {
      sim_bunchCrossing_branch->GetEntry(index);
    } else {
      printf("branch sim_bunchCrossing_branch does not exist!\n");
      exit(1);
    }
    sim_bunchCrossing_isLoaded = true;
  }
  return *sim_bunchCrossing_;
}
const std::vector<float> &Trktree::see_stateCcov45() {
  if (not see_stateCcov45_isLoaded) {
    if (see_stateCcov45_branch != 0) {
      see_stateCcov45_branch->GetEntry(index);
    } else {
      printf("branch see_stateCcov45_branch does not exist!\n");
      exit(1);
    }
    see_stateCcov45_isLoaded = true;
  }
  return *see_stateCcov45_;
}
const std::vector<unsigned short> &Trktree::ph2_isStack() {
  if (not ph2_isStack_isLoaded) {
    if (ph2_isStack_branch != 0) {
      ph2_isStack_branch->GetEntry(index);
    } else {
      printf("branch ph2_isStack_branch does not exist!\n");
      exit(1);
    }
    ph2_isStack_isLoaded = true;
  }
  return *ph2_isStack_;
}
const std::vector<std::vector<float> > &Trktree::sim_trkShareFrac() {
  if (not sim_trkShareFrac_isLoaded) {
    if (sim_trkShareFrac_branch != 0) {
      sim_trkShareFrac_branch->GetEntry(index);
    } else {
      printf("branch sim_trkShareFrac_branch does not exist!\n");
      exit(1);
    }
    sim_trkShareFrac_isLoaded = true;
  }
  return *sim_trkShareFrac_;
}
const std::vector<std::vector<float> > &Trktree::trk_simTrkShareFrac() {
  if (not trk_simTrkShareFrac_isLoaded) {
    if (trk_simTrkShareFrac_branch != 0) {
      trk_simTrkShareFrac_branch->GetEntry(index);
    } else {
      printf("branch trk_simTrkShareFrac_branch does not exist!\n");
      exit(1);
    }
    trk_simTrkShareFrac_isLoaded = true;
  }
  return *trk_simTrkShareFrac_;
}
const std::vector<float> &Trktree::sim_phi() {
  if (not sim_phi_isLoaded) {
    if (sim_phi_branch != 0) {
      sim_phi_branch->GetEntry(index);
    } else {
      printf("branch sim_phi_branch does not exist!\n");
      exit(1);
    }
    sim_phi_isLoaded = true;
  }
  return *sim_phi_;
}
const std::vector<unsigned short> &Trktree::inv_side() {
  if (not inv_side_isLoaded) {
    if (inv_side_branch != 0) {
      inv_side_branch->GetEntry(index);
    } else {
      printf("branch inv_side_branch does not exist!\n");
      exit(1);
    }
    inv_side_isLoaded = true;
  }
  return *inv_side_;
}
const std::vector<short> &Trktree::vtx_fake() {
  if (not vtx_fake_isLoaded) {
    if (vtx_fake_branch != 0) {
      vtx_fake_branch->GetEntry(index);
    } else {
      printf("branch vtx_fake_branch does not exist!\n");
      exit(1);
    }
    vtx_fake_isLoaded = true;
  }
  return *vtx_fake_;
}
const std::vector<unsigned int> &Trktree::trk_nInactive() {
  if (not trk_nInactive_isLoaded) {
    if (trk_nInactive_branch != 0) {
      trk_nInactive_branch->GetEntry(index);
    } else {
      printf("branch trk_nInactive_branch does not exist!\n");
      exit(1);
    }
    trk_nInactive_isLoaded = true;
  }
  return *trk_nInactive_;
}
const std::vector<unsigned int> &Trktree::trk_nPixelLay() {
  if (not trk_nPixelLay_isLoaded) {
    if (trk_nPixelLay_branch != 0) {
      trk_nPixelLay_branch->GetEntry(index);
    } else {
      printf("branch trk_nPixelLay_branch does not exist!\n");
      exit(1);
    }
    trk_nPixelLay_isLoaded = true;
  }
  return *trk_nPixelLay_;
}
const std::vector<float> &Trktree::ph2_bbxi() {
  if (not ph2_bbxi_isLoaded) {
    if (ph2_bbxi_branch != 0) {
      ph2_bbxi_branch->GetEntry(index);
    } else {
      printf("branch ph2_bbxi_branch does not exist!\n");
      exit(1);
    }
    ph2_bbxi_isLoaded = true;
  }
  return *ph2_bbxi_;
}
const std::vector<float> &Trktree::vtx_xErr() {
  if (not vtx_xErr_isLoaded) {
    if (vtx_xErr_branch != 0) {
      vtx_xErr_branch->GetEntry(index);
    } else {
      printf("branch vtx_xErr_branch does not exist!\n");
      exit(1);
    }
    vtx_xErr_isLoaded = true;
  }
  return *vtx_xErr_;
}
const std::vector<float> &Trktree::see_stateCcov25() {
  if (not see_stateCcov25_isLoaded) {
    if (see_stateCcov25_branch != 0) {
      see_stateCcov25_branch->GetEntry(index);
    } else {
      printf("branch see_stateCcov25_branch does not exist!\n");
      exit(1);
    }
    see_stateCcov25_isLoaded = true;
  }
  return *see_stateCcov25_;
}
const std::vector<int> &Trktree::sim_parentVtxIdx() {
  if (not sim_parentVtxIdx_isLoaded) {
    if (sim_parentVtxIdx_branch != 0) {
      sim_parentVtxIdx_branch->GetEntry(index);
    } else {
      printf("branch sim_parentVtxIdx_branch does not exist!\n");
      exit(1);
    }
    sim_parentVtxIdx_isLoaded = true;
  }
  return *sim_parentVtxIdx_;
}
const std::vector<float> &Trktree::see_stateCcov23() {
  if (not see_stateCcov23_isLoaded) {
    if (see_stateCcov23_branch != 0) {
      see_stateCcov23_branch->GetEntry(index);
    } else {
      printf("branch see_stateCcov23_branch does not exist!\n");
      exit(1);
    }
    see_stateCcov23_isLoaded = true;
  }
  return *see_stateCcov23_;
}
const std::vector<ULong64_t> &Trktree::trk_algoMask() {
  if (not trk_algoMask_isLoaded) {
    if (trk_algoMask_branch != 0) {
      trk_algoMask_branch->GetEntry(index);
    } else {
      printf("branch trk_algoMask_branch does not exist!\n");
      exit(1);
    }
    trk_algoMask_isLoaded = true;
  }
  return *trk_algoMask_;
}
const std::vector<std::vector<int> > &Trktree::trk_simTrkIdx() {
  if (not trk_simTrkIdx_isLoaded) {
    if (trk_simTrkIdx_branch != 0) {
      trk_simTrkIdx_branch->GetEntry(index);
    } else {
      printf("branch trk_simTrkIdx_branch does not exist!\n");
      exit(1);
    }
    trk_simTrkIdx_isLoaded = true;
  }
  return *trk_simTrkIdx_;
}
const std::vector<float> &Trktree::see_phiErr() {
  if (not see_phiErr_isLoaded) {
    if (see_phiErr_branch != 0) {
      see_phiErr_branch->GetEntry(index);
    } else {
      printf("branch see_phiErr_branch does not exist!\n");
      exit(1);
    }
    see_phiErr_isLoaded = true;
  }
  return *see_phiErr_;
}
const std::vector<float> &Trktree::trk_cotTheta() {
  if (not trk_cotTheta_isLoaded) {
    if (trk_cotTheta_branch != 0) {
      trk_cotTheta_branch->GetEntry(index);
    } else {
      printf("branch trk_cotTheta_branch does not exist!\n");
      exit(1);
    }
    trk_cotTheta_isLoaded = true;
  }
  return *trk_cotTheta_;
}
const std::vector<unsigned int> &Trktree::see_algo() {
  if (not see_algo_isLoaded) {
    if (see_algo_branch != 0) {
      see_algo_branch->GetEntry(index);
    } else {
      printf("branch see_algo_branch does not exist!\n");
      exit(1);
    }
    see_algo_isLoaded = true;
  }
  return *see_algo_;
}
const std::vector<unsigned short> &Trktree::simhit_module() {
  if (not simhit_module_isLoaded) {
    if (simhit_module_branch != 0) {
      simhit_module_branch->GetEntry(index);
    } else {
      printf("branch simhit_module_branch does not exist!\n");
      exit(1);
    }
    simhit_module_isLoaded = true;
  }
  return *simhit_module_;
}
const std::vector<std::vector<int> > &Trktree::simvtx_daughterSimIdx() {
  if (not simvtx_daughterSimIdx_isLoaded) {
    if (simvtx_daughterSimIdx_branch != 0) {
      simvtx_daughterSimIdx_branch->GetEntry(index);
    } else {
      printf("branch simvtx_daughterSimIdx_branch does not exist!\n");
      exit(1);
    }
    simvtx_daughterSimIdx_isLoaded = true;
  }
  return *simvtx_daughterSimIdx_;
}
const std::vector<float> &Trktree::vtx_x() {
  if (not vtx_x_isLoaded) {
    if (vtx_x_branch != 0) {
      vtx_x_branch->GetEntry(index);
    } else {
      printf("branch vtx_x_branch does not exist!\n");
      exit(1);
    }
    vtx_x_isLoaded = true;
  }
  return *vtx_x_;
}
const std::vector<int> &Trktree::trk_seedIdx() {
  if (not trk_seedIdx_isLoaded) {
    if (trk_seedIdx_branch != 0) {
      trk_seedIdx_branch->GetEntry(index);
    } else {
      printf("branch trk_seedIdx_branch does not exist!\n");
      exit(1);
    }
    trk_seedIdx_isLoaded = true;
  }
  return *trk_seedIdx_;
}
const std::vector<float> &Trktree::simhit_y() {
  if (not simhit_y_isLoaded) {
    if (simhit_y_branch != 0) {
      simhit_y_branch->GetEntry(index);
    } else {
      printf("branch simhit_y_branch does not exist!\n");
      exit(1);
    }
    simhit_y_isLoaded = true;
  }
  return *simhit_y_;
}
const std::vector<unsigned short> &Trktree::inv_layer() {
  if (not inv_layer_isLoaded) {
    if (inv_layer_branch != 0) {
      inv_layer_branch->GetEntry(index);
    } else {
      printf("branch inv_layer_branch does not exist!\n");
      exit(1);
    }
    inv_layer_isLoaded = true;
  }
  return *inv_layer_;
}
const std::vector<unsigned int> &Trktree::trk_nLostLay() {
  if (not trk_nLostLay_isLoaded) {
    if (trk_nLostLay_branch != 0) {
      trk_nLostLay_branch->GetEntry(index);
    } else {
      printf("branch trk_nLostLay_branch does not exist!\n");
      exit(1);
    }
    trk_nLostLay_isLoaded = true;
  }
  return *trk_nLostLay_;
}
const std::vector<unsigned short> &Trktree::ph2_isLower() {
  if (not ph2_isLower_isLoaded) {
    if (ph2_isLower_branch != 0) {
      ph2_isLower_branch->GetEntry(index);
    } else {
      printf("branch ph2_isLower_branch does not exist!\n");
      exit(1);
    }
    ph2_isLower_isLoaded = true;
  }
  return *ph2_isLower_;
}
const std::vector<unsigned short> &Trktree::pix_side() {
  if (not pix_side_isLoaded) {
    if (pix_side_branch != 0) {
      pix_side_branch->GetEntry(index);
    } else {
      printf("branch pix_side_branch does not exist!\n");
      exit(1);
    }
    pix_side_isLoaded = true;
  }
  return *pix_side_;
}
const std::vector<unsigned short> &Trktree::inv_isLower() {
  if (not inv_isLower_isLoaded) {
    if (inv_isLower_branch != 0) {
      inv_isLower_branch->GetEntry(index);
    } else {
      printf("branch inv_isLower_branch does not exist!\n");
      exit(1);
    }
    inv_isLower_isLoaded = true;
  }
  return *inv_isLower_;
}
const std::vector<std::vector<int> > &Trktree::ph2_trkIdx() {
  if (not ph2_trkIdx_isLoaded) {
    if (ph2_trkIdx_branch != 0) {
      ph2_trkIdx_branch->GetEntry(index);
    } else {
      printf("branch ph2_trkIdx_branch does not exist!\n");
      exit(1);
    }
    ph2_trkIdx_isLoaded = true;
  }
  return *ph2_trkIdx_;
}
const std::vector<unsigned int> &Trktree::sim_nValid() {
  if (not sim_nValid_isLoaded) {
    if (sim_nValid_branch != 0) {
      sim_nValid_branch->GetEntry(index);
    } else {
      printf("branch sim_nValid_branch does not exist!\n");
      exit(1);
    }
    sim_nValid_isLoaded = true;
  }
  return *sim_nValid_;
}
const std::vector<int> &Trktree::simhit_simTrkIdx() {
  if (not simhit_simTrkIdx_isLoaded) {
    if (simhit_simTrkIdx_branch != 0) {
      simhit_simTrkIdx_branch->GetEntry(index);
    } else {
      printf("branch simhit_simTrkIdx_branch does not exist!\n");
      exit(1);
    }
    simhit_simTrkIdx_isLoaded = true;
  }
  return *simhit_simTrkIdx_;
}
const std::vector<unsigned short> &Trktree::see_nCands() {
  if (not see_nCands_isLoaded) {
    if (see_nCands_branch != 0) {
      see_nCands_branch->GetEntry(index);
    } else {
      printf("branch see_nCands_branch does not exist!\n");
      exit(1);
    }
    see_nCands_isLoaded = true;
  }
  return *see_nCands_;
}
const std::vector<int> &Trktree::see_bestSimTrkIdx() {
  if (not see_bestSimTrkIdx_isLoaded) {
    if (see_bestSimTrkIdx_branch != 0) {
      see_bestSimTrkIdx_branch->GetEntry(index);
    } else {
      printf("branch see_bestSimTrkIdx_branch does not exist!\n");
      exit(1);
    }
    see_bestSimTrkIdx_isLoaded = true;
  }
  return *see_bestSimTrkIdx_;
}
const std::vector<float> &Trktree::vtx_yErr() {
  if (not vtx_yErr_isLoaded) {
    if (vtx_yErr_branch != 0) {
      vtx_yErr_branch->GetEntry(index);
    } else {
      printf("branch vtx_yErr_branch does not exist!\n");
      exit(1);
    }
    vtx_yErr_isLoaded = true;
  }
  return *vtx_yErr_;
}
const std::vector<float> &Trktree::trk_dzPV() {
  if (not trk_dzPV_isLoaded) {
    if (trk_dzPV_branch != 0) {
      trk_dzPV_branch->GetEntry(index);
    } else {
      printf("branch trk_dzPV_branch does not exist!\n");
      exit(1);
    }
    trk_dzPV_isLoaded = true;
  }
  return *trk_dzPV_;
}
const std::vector<float> &Trktree::ph2_xy() {
  if (not ph2_xy_isLoaded) {
    if (ph2_xy_branch != 0) {
      ph2_xy_branch->GetEntry(index);
    } else {
      printf("branch ph2_xy_branch does not exist!\n");
      exit(1);
    }
    ph2_xy_isLoaded = true;
  }
  return *ph2_xy_;
}
const std::vector<unsigned short> &Trktree::inv_module() {
  if (not inv_module_isLoaded) {
    if (inv_module_branch != 0) {
      inv_module_branch->GetEntry(index);
    } else {
      printf("branch inv_module_branch does not exist!\n");
      exit(1);
    }
    inv_module_isLoaded = true;
  }
  return *inv_module_;
}
const std::vector<float> &Trktree::see_stateCcov55() {
  if (not see_stateCcov55_isLoaded) {
    if (see_stateCcov55_branch != 0) {
      see_stateCcov55_branch->GetEntry(index);
    } else {
      printf("branch see_stateCcov55_branch does not exist!\n");
      exit(1);
    }
    see_stateCcov55_isLoaded = true;
  }
  return *see_stateCcov55_;
}
const std::vector<unsigned short> &Trktree::pix_panel() {
  if (not pix_panel_isLoaded) {
    if (pix_panel_branch != 0) {
      pix_panel_branch->GetEntry(index);
    } else {
      printf("branch pix_panel_branch does not exist!\n");
      exit(1);
    }
    pix_panel_isLoaded = true;
  }
  return *pix_panel_;
}
const std::vector<unsigned short> &Trktree::inv_ladder() {
  if (not inv_ladder_isLoaded) {
    if (inv_ladder_branch != 0) {
      inv_ladder_branch->GetEntry(index);
    } else {
      printf("branch inv_ladder_branch does not exist!\n");
      exit(1);
    }
    inv_ladder_isLoaded = true;
  }
  return *inv_ladder_;
}
const std::vector<float> &Trktree::ph2_xx() {
  if (not ph2_xx_isLoaded) {
    if (ph2_xx_branch != 0) {
      ph2_xx_branch->GetEntry(index);
    } else {
      printf("branch ph2_xx_branch does not exist!\n");
      exit(1);
    }
    ph2_xx_isLoaded = true;
  }
  return *ph2_xx_;
}
const std::vector<float> &Trktree::sim_pca_cotTheta() {
  if (not sim_pca_cotTheta_isLoaded) {
    if (sim_pca_cotTheta_branch != 0) {
      sim_pca_cotTheta_branch->GetEntry(index);
    } else {
      printf("branch sim_pca_cotTheta_branch does not exist!\n");
      exit(1);
    }
    sim_pca_cotTheta_isLoaded = true;
  }
  return *sim_pca_cotTheta_;
}
const std::vector<int> &Trktree::simpv_idx() {
  if (not simpv_idx_isLoaded) {
    if (simpv_idx_branch != 0) {
      simpv_idx_branch->GetEntry(index);
    } else {
      printf("branch simpv_idx_branch does not exist!\n");
      exit(1);
    }
    simpv_idx_isLoaded = true;
  }
  return *simpv_idx_;
}
const std::vector<float> &Trktree::trk_inner_pz() {
  if (not trk_inner_pz_isLoaded) {
    if (trk_inner_pz_branch != 0) {
      trk_inner_pz_branch->GetEntry(index);
    } else {
      printf("branch trk_inner_pz_branch does not exist!\n");
      exit(1);
    }
    trk_inner_pz_isLoaded = true;
  }
  return *trk_inner_pz_;
}
const std::vector<float> &Trktree::see_chi2() {
  if (not see_chi2_isLoaded) {
    if (see_chi2_branch != 0) {
      see_chi2_branch->GetEntry(index);
    } else {
      printf("branch see_chi2_branch does not exist!\n");
      exit(1);
    }
    see_chi2_isLoaded = true;
  }
  return *see_chi2_;
}
const std::vector<float> &Trktree::see_stateCcov35() {
  if (not see_stateCcov35_isLoaded) {
    if (see_stateCcov35_branch != 0) {
      see_stateCcov35_branch->GetEntry(index);
    } else {
      printf("branch see_stateCcov35_branch does not exist!\n");
      exit(1);
    }
    see_stateCcov35_isLoaded = true;
  }
  return *see_stateCcov35_;
}
const std::vector<float> &Trktree::see_stateCcov33() {
  if (not see_stateCcov33_isLoaded) {
    if (see_stateCcov33_branch != 0) {
      see_stateCcov33_branch->GetEntry(index);
    } else {
      printf("branch see_stateCcov33_branch does not exist!\n");
      exit(1);
    }
    see_stateCcov33_isLoaded = true;
  }
  return *see_stateCcov33_;
}
const std::vector<unsigned int> &Trktree::inv_detId() {
  if (not inv_detId_isLoaded) {
    if (inv_detId_branch != 0) {
      inv_detId_branch->GetEntry(index);
    } else {
      printf("branch inv_detId_branch does not exist!\n");
      exit(1);
    }
    inv_detId_isLoaded = true;
  }
  return *inv_detId_;
}
const std::vector<unsigned int> &Trktree::see_offset() {
  if (not see_offset_isLoaded) {
    if (see_offset_branch != 0) {
      see_offset_branch->GetEntry(index);
    } else {
      printf("branch see_offset_branch does not exist!\n");
      exit(1);
    }
    see_offset_isLoaded = true;
  }
  return *see_offset_;
}
const std::vector<unsigned int> &Trktree::sim_nLay() {
  if (not sim_nLay_isLoaded) {
    if (sim_nLay_branch != 0) {
      sim_nLay_branch->GetEntry(index);
    } else {
      printf("branch sim_nLay_branch does not exist!\n");
      exit(1);
    }
    sim_nLay_isLoaded = true;
  }
  return *sim_nLay_;
}
const std::vector<std::vector<int> > &Trktree::sim_simHitIdx() {
  if (not sim_simHitIdx_isLoaded) {
    if (sim_simHitIdx_branch != 0) {
      sim_simHitIdx_branch->GetEntry(index);
    } else {
      printf("branch sim_simHitIdx_branch does not exist!\n");
      exit(1);
    }
    sim_simHitIdx_isLoaded = true;
  }
  return *sim_simHitIdx_;
}
const std::vector<unsigned short> &Trktree::simhit_isUpper() {
  if (not simhit_isUpper_isLoaded) {
    if (simhit_isUpper_branch != 0) {
      simhit_isUpper_branch->GetEntry(index);
    } else {
      printf("branch simhit_isUpper_branch does not exist!\n");
      exit(1);
    }
    simhit_isUpper_isLoaded = true;
  }
  return *simhit_isUpper_;
}
const std::vector<float> &Trktree::see_stateCcov00() {
  if (not see_stateCcov00_isLoaded) {
    if (see_stateCcov00_branch != 0) {
      see_stateCcov00_branch->GetEntry(index);
    } else {
      printf("branch see_stateCcov00_branch does not exist!\n");
      exit(1);
    }
    see_stateCcov00_isLoaded = true;
  }
  return *see_stateCcov00_;
}
const std::vector<unsigned short> &Trktree::see_stopReason() {
  if (not see_stopReason_isLoaded) {
    if (see_stopReason_branch != 0) {
      see_stopReason_branch->GetEntry(index);
    } else {
      printf("branch see_stopReason_branch does not exist!\n");
      exit(1);
    }
    see_stopReason_isLoaded = true;
  }
  return *see_stopReason_;
}
const std::vector<short> &Trktree::vtx_valid() {
  if (not vtx_valid_isLoaded) {
    if (vtx_valid_branch != 0) {
      vtx_valid_branch->GetEntry(index);
    } else {
      printf("branch vtx_valid_branch does not exist!\n");
      exit(1);
    }
    vtx_valid_isLoaded = true;
  }
  return *vtx_valid_;
}
const unsigned int &Trktree::lumi() {
  if (not lumi_isLoaded) {
    if (lumi_branch != 0) {
      lumi_branch->GetEntry(index);
    } else {
      printf("branch lumi_branch does not exist!\n");
      exit(1);
    }
    lumi_isLoaded = true;
  }
  return lumi_;
}
const std::vector<float> &Trktree::trk_refpoint_x() {
  if (not trk_refpoint_x_isLoaded) {
    if (trk_refpoint_x_branch != 0) {
      trk_refpoint_x_branch->GetEntry(index);
    } else {
      printf("branch trk_refpoint_x_branch does not exist!\n");
      exit(1);
    }
    trk_refpoint_x_isLoaded = true;
  }
  return *trk_refpoint_x_;
}
const std::vector<float> &Trktree::trk_refpoint_y() {
  if (not trk_refpoint_y_isLoaded) {
    if (trk_refpoint_y_branch != 0) {
      trk_refpoint_y_branch->GetEntry(index);
    } else {
      printf("branch trk_refpoint_y_branch does not exist!\n");
      exit(1);
    }
    trk_refpoint_y_isLoaded = true;
  }
  return *trk_refpoint_y_;
}
const std::vector<float> &Trktree::trk_refpoint_z() {
  if (not trk_refpoint_z_isLoaded) {
    if (trk_refpoint_z_branch != 0) {
      trk_refpoint_z_branch->GetEntry(index);
    } else {
      printf("branch trk_refpoint_z_branch does not exist!\n");
      exit(1);
    }
    trk_refpoint_z_isLoaded = true;
  }
  return *trk_refpoint_z_;
}
const std::vector<unsigned int> &Trktree::sim_n3DLay() {
  if (not sim_n3DLay_isLoaded) {
    if (sim_n3DLay_branch != 0) {
      sim_n3DLay_branch->GetEntry(index);
    } else {
      printf("branch sim_n3DLay_branch does not exist!\n");
      exit(1);
    }
    sim_n3DLay_isLoaded = true;
  }
  return *sim_n3DLay_;
}
const std::vector<unsigned int> &Trktree::see_nPhase2OT() {
  if (not see_nPhase2OT_isLoaded) {
    if (see_nPhase2OT_branch != 0) {
      see_nPhase2OT_branch->GetEntry(index);
    } else {
      printf("branch see_nPhase2OT_branch does not exist!\n");
      exit(1);
    }
    see_nPhase2OT_isLoaded = true;
  }
  return *see_nPhase2OT_;
}
const std::vector<float> &Trktree::trk_bestFromFirstHitSimTrkShareFracSimDenom() {
  if (not trk_bestFromFirstHitSimTrkShareFracSimDenom_isLoaded) {
    if (trk_bestFromFirstHitSimTrkShareFracSimDenom_branch != 0) {
      trk_bestFromFirstHitSimTrkShareFracSimDenom_branch->GetEntry(index);
    } else {
      printf("branch trk_bestFromFirstHitSimTrkShareFracSimDenom_branch does not exist!\n");
      exit(1);
    }
    trk_bestFromFirstHitSimTrkShareFracSimDenom_isLoaded = true;
  }
  return *trk_bestFromFirstHitSimTrkShareFracSimDenom_;
}
const std::vector<float> &Trktree::ph2_yy() {
  if (not ph2_yy_isLoaded) {
    if (ph2_yy_branch != 0) {
      ph2_yy_branch->GetEntry(index);
    } else {
      printf("branch ph2_yy_branch does not exist!\n");
      exit(1);
    }
    ph2_yy_isLoaded = true;
  }
  return *ph2_yy_;
}
const std::vector<float> &Trktree::ph2_yz() {
  if (not ph2_yz_isLoaded) {
    if (ph2_yz_branch != 0) {
      ph2_yz_branch->GetEntry(index);
    } else {
      printf("branch ph2_yz_branch does not exist!\n");
      exit(1);
    }
    ph2_yz_isLoaded = true;
  }
  return *ph2_yz_;
}
const std::vector<unsigned short> &Trktree::inv_blade() {
  if (not inv_blade_isLoaded) {
    if (inv_blade_branch != 0) {
      inv_blade_branch->GetEntry(index);
    } else {
      printf("branch inv_blade_branch does not exist!\n");
      exit(1);
    }
    inv_blade_isLoaded = true;
  }
  return *inv_blade_;
}
const std::vector<float> &Trktree::trk_ptErr() {
  if (not trk_ptErr_isLoaded) {
    if (trk_ptErr_branch != 0) {
      trk_ptErr_branch->GetEntry(index);
    } else {
      printf("branch trk_ptErr_branch does not exist!\n");
      exit(1);
    }
    trk_ptErr_isLoaded = true;
  }
  return *trk_ptErr_;
}
const std::vector<float> &Trktree::pix_zx() {
  if (not pix_zx_isLoaded) {
    if (pix_zx_branch != 0) {
      pix_zx_branch->GetEntry(index);
    } else {
      printf("branch pix_zx_branch does not exist!\n");
      exit(1);
    }
    pix_zx_isLoaded = true;
  }
  return *pix_zx_;
}
const std::vector<float> &Trktree::simvtx_z() {
  if (not simvtx_z_isLoaded) {
    if (simvtx_z_branch != 0) {
      simvtx_z_branch->GetEntry(index);
    } else {
      printf("branch simvtx_z_branch does not exist!\n");
      exit(1);
    }
    simvtx_z_isLoaded = true;
  }
  return *simvtx_z_;
}
const std::vector<unsigned int> &Trktree::sim_nTrackerHits() {
  if (not sim_nTrackerHits_isLoaded) {
    if (sim_nTrackerHits_branch != 0) {
      sim_nTrackerHits_branch->GetEntry(index);
    } else {
      printf("branch sim_nTrackerHits_branch does not exist!\n");
      exit(1);
    }
    sim_nTrackerHits_isLoaded = true;
  }
  return *sim_nTrackerHits_;
}
const std::vector<unsigned short> &Trktree::ph2_subdet() {
  if (not ph2_subdet_isLoaded) {
    if (ph2_subdet_branch != 0) {
      ph2_subdet_branch->GetEntry(index);
    } else {
      printf("branch ph2_subdet_branch does not exist!\n");
      exit(1);
    }
    ph2_subdet_isLoaded = true;
  }
  return *ph2_subdet_;
}
const std::vector<float> &Trktree::see_stateTrajPx() {
  if (not see_stateTrajPx_isLoaded) {
    if (see_stateTrajPx_branch != 0) {
      see_stateTrajPx_branch->GetEntry(index);
    } else {
      printf("branch see_stateTrajPx_branch does not exist!\n");
      exit(1);
    }
    see_stateTrajPx_isLoaded = true;
  }
  return *see_stateTrajPx_;
}
const std::vector<std::vector<int> > &Trktree::simhit_hitIdx() {
  if (not simhit_hitIdx_isLoaded) {
    if (simhit_hitIdx_branch != 0) {
      simhit_hitIdx_branch->GetEntry(index);
    } else {
      printf("branch simhit_hitIdx_branch does not exist!\n");
      exit(1);
    }
    simhit_hitIdx_isLoaded = true;
  }
  return *simhit_hitIdx_;
}
const std::vector<unsigned short> &Trktree::simhit_ladder() {
  if (not simhit_ladder_isLoaded) {
    if (simhit_ladder_branch != 0) {
      simhit_ladder_branch->GetEntry(index);
    } else {
      printf("branch simhit_ladder_branch does not exist!\n");
      exit(1);
    }
    simhit_ladder_isLoaded = true;
  }
  return *simhit_ladder_;
}
const std::vector<unsigned short> &Trktree::ph2_layer() {
  if (not ph2_layer_isLoaded) {
    if (ph2_layer_branch != 0) {
      ph2_layer_branch->GetEntry(index);
    } else {
      printf("branch ph2_layer_branch does not exist!\n");
      exit(1);
    }
    ph2_layer_isLoaded = true;
  }
  return *ph2_layer_;
}
const std::vector<float> &Trktree::see_phi() {
  if (not see_phi_isLoaded) {
    if (see_phi_branch != 0) {
      see_phi_branch->GetEntry(index);
    } else {
      printf("branch see_phi_branch does not exist!\n");
      exit(1);
    }
    see_phi_isLoaded = true;
  }
  return *see_phi_;
}
const std::vector<float> &Trktree::trk_nChi2_1Dmod() {
  if (not trk_nChi2_1Dmod_isLoaded) {
    if (trk_nChi2_1Dmod_branch != 0) {
      trk_nChi2_1Dmod_branch->GetEntry(index);
    } else {
      printf("branch trk_nChi2_1Dmod_branch does not exist!\n");
      exit(1);
    }
    trk_nChi2_1Dmod_isLoaded = true;
  }
  return *trk_nChi2_1Dmod_;
}
const std::vector<float> &Trktree::trk_inner_py() {
  if (not trk_inner_py_isLoaded) {
    if (trk_inner_py_branch != 0) {
      trk_inner_py_branch->GetEntry(index);
    } else {
      printf("branch trk_inner_py_branch does not exist!\n");
      exit(1);
    }
    trk_inner_py_isLoaded = true;
  }
  return *trk_inner_py_;
}
const std::vector<float> &Trktree::trk_inner_px() {
  if (not trk_inner_px_isLoaded) {
    if (trk_inner_px_branch != 0) {
      trk_inner_px_branch->GetEntry(index);
    } else {
      printf("branch trk_inner_px_branch does not exist!\n");
      exit(1);
    }
    trk_inner_px_isLoaded = true;
  }
  return *trk_inner_px_;
}
const std::vector<float> &Trktree::trk_dxyErr() {
  if (not trk_dxyErr_isLoaded) {
    if (trk_dxyErr_branch != 0) {
      trk_dxyErr_branch->GetEntry(index);
    } else {
      printf("branch trk_dxyErr_branch does not exist!\n");
      exit(1);
    }
    trk_dxyErr_isLoaded = true;
  }
  return *trk_dxyErr_;
}
const std::vector<unsigned int> &Trktree::sim_nPixelLay() {
  if (not sim_nPixelLay_isLoaded) {
    if (sim_nPixelLay_branch != 0) {
      sim_nPixelLay_branch->GetEntry(index);
    } else {
      printf("branch sim_nPixelLay_branch does not exist!\n");
      exit(1);
    }
    sim_nPixelLay_isLoaded = true;
  }
  return *sim_nPixelLay_;
}
const std::vector<unsigned int> &Trktree::see_nValid() {
  if (not see_nValid_isLoaded) {
    if (see_nValid_branch != 0) {
      see_nValid_branch->GetEntry(index);
    } else {
      printf("branch see_nValid_branch does not exist!\n");
      exit(1);
    }
    see_nValid_isLoaded = true;
  }
  return *see_nValid_;
}
const std::vector<float> &Trktree::trk_inner_pt() {
  if (not trk_inner_pt_isLoaded) {
    if (trk_inner_pt_branch != 0) {
      trk_inner_pt_branch->GetEntry(index);
    } else {
      printf("branch trk_inner_pt_branch does not exist!\n");
      exit(1);
    }
    trk_inner_pt_isLoaded = true;
  }
  return *trk_inner_pt_;
}
const std::vector<float> &Trktree::see_stateTrajGlbPy() {
  if (not see_stateTrajGlbPy_isLoaded) {
    if (see_stateTrajGlbPy_branch != 0) {
      see_stateTrajGlbPy_branch->GetEntry(index);
    } else {
      printf("branch see_stateTrajGlbPy_branch does not exist!\n");
      exit(1);
    }
    see_stateTrajGlbPy_isLoaded = true;
  }
  return *see_stateTrajGlbPy_;
}
void Trktree::progress(int nEventsTotal, int nEventsChain) {
  int period = 1000;
  if (nEventsTotal % 1000 == 0) {
    if (isatty(1)) {
      if ((nEventsChain - nEventsTotal) > period) {
        float frac = (float)nEventsTotal / (nEventsChain * 0.01);
        printf(
            "\015\033[32m ---> \033[1m\033[31m%4.1f%%"
            "\033[0m\033[32m <---\033[0m\015",
            frac);
        fflush(stdout);
      } else {
        printf(
            "\015\033[32m ---> \033[1m\033[31m%4.1f%%"
            "\033[0m\033[32m <---\033[0m\015",
            100.);
        std::cout << std::endl;
      }
    }
  }
}
namespace tas {
  const std::vector<float> &see_stateCcov01() { return trk.see_stateCcov01(); }
  const std::vector<unsigned short> &simhit_rod() { return trk.simhit_rod(); }
  const std::vector<float> &trk_phi() { return trk.trk_phi(); }
  const float &bsp_x() { return trk.bsp_x(); }
  const std::vector<float> &see_stateCcov05() { return trk.see_stateCcov05(); }
  const std::vector<float> &see_stateCcov04() { return trk.see_stateCcov04(); }
  const std::vector<float> &trk_dxyPV() { return trk.trk_dxyPV(); }
  const std::vector<float> &simhit_tof() { return trk.simhit_tof(); }
  const std::vector<int> &sim_event() { return trk.sim_event(); }
  const std::vector<unsigned short> &simhit_isStack() { return trk.simhit_isStack(); }
  const std::vector<float> &trk_dz() { return trk.trk_dz(); }
  const std::vector<float> &see_stateCcov03() { return trk.see_stateCcov03(); }
  const std::vector<float> &sim_eta() { return trk.sim_eta(); }
  const std::vector<unsigned int> &simvtx_processType() { return trk.simvtx_processType(); }
  const std::vector<float> &pix_radL() { return trk.pix_radL(); }
  const std::vector<float> &see_stateCcov02() { return trk.see_stateCcov02(); }
  const std::vector<unsigned int> &see_nGlued() { return trk.see_nGlued(); }
  const std::vector<int> &trk_bestSimTrkIdx() { return trk.trk_bestSimTrkIdx(); }
  const std::vector<float> &see_stateTrajGlbPz() { return trk.see_stateTrajGlbPz(); }
  const std::vector<float> &pix_yz() { return trk.pix_yz(); }
  const std::vector<float> &pix_yy() { return trk.pix_yy(); }
  const std::vector<short> &simhit_process() { return trk.simhit_process(); }
  const std::vector<float> &see_stateCcov34() { return trk.see_stateCcov34(); }
  const std::vector<unsigned int> &trk_nInnerLost() { return trk.trk_nInnerLost(); }
  const std::vector<float> &see_py() { return trk.see_py(); }
  const std::vector<std::vector<int> > &sim_trkIdx() { return trk.sim_trkIdx(); }
  const std::vector<unsigned int> &trk_nLost() { return trk.trk_nLost(); }
  const std::vector<short> &pix_isBarrel() { return trk.pix_isBarrel(); }
  const std::vector<float> &see_dxyErr() { return trk.see_dxyErr(); }
  const std::vector<unsigned int> &simhit_detId() { return trk.simhit_detId(); }
  const std::vector<unsigned short> &simhit_subdet() { return trk.simhit_subdet(); }
  const std::vector<std::vector<int> > &see_hitIdx() { return trk.see_hitIdx(); }
  const std::vector<float> &see_pt() { return trk.see_pt(); }
  const std::vector<unsigned int> &ph2_detId() { return trk.ph2_detId(); }
  const std::vector<unsigned int> &trk_nStripLay() { return trk.trk_nStripLay(); }
  const std::vector<int> &see_bestFromFirstHitSimTrkIdx() { return trk.see_bestFromFirstHitSimTrkIdx(); }
  const std::vector<float> &sim_pca_pt() { return trk.sim_pca_pt(); }
  const std::vector<int> &see_trkIdx() { return trk.see_trkIdx(); }
  const std::vector<unsigned int> &trk_nCluster() { return trk.trk_nCluster(); }
  const std::vector<float> &trk_bestFromFirstHitSimTrkNChi2() { return trk.trk_bestFromFirstHitSimTrkNChi2(); }
  const std::vector<short> &trk_isHP() { return trk.trk_isHP(); }
  const std::vector<std::vector<int> > &simhit_hitType() { return trk.simhit_hitType(); }
  const std::vector<unsigned short> &ph2_isUpper() { return trk.ph2_isUpper(); }
  const std::vector<unsigned int> &see_nStrip() { return trk.see_nStrip(); }
  const std::vector<float> &trk_bestSimTrkShareFracSimClusterDenom() {
    return trk.trk_bestSimTrkShareFracSimClusterDenom();
  }
  const std::vector<unsigned short> &simhit_side() { return trk.simhit_side(); }
  const std::vector<float> &simhit_x() { return trk.simhit_x(); }
  const std::vector<int> &see_q() { return trk.see_q(); }
  const std::vector<float> &simhit_z() { return trk.simhit_z(); }
  const std::vector<float> &sim_pca_lambda() { return trk.sim_pca_lambda(); }
  const std::vector<int> &sim_q() { return trk.sim_q(); }
  const std::vector<float> &pix_bbxi() { return trk.pix_bbxi(); }
  const std::vector<unsigned short> &ph2_order() { return trk.ph2_order(); }
  const std::vector<unsigned short> &ph2_module() { return trk.ph2_module(); }
  const std::vector<unsigned short> &inv_order() { return trk.inv_order(); }
  const std::vector<float> &trk_dzErr() { return trk.trk_dzErr(); }
  const std::vector<unsigned int> &trk_nInnerInactive() { return trk.trk_nInnerInactive(); }
  const std::vector<short> &see_fitok() { return trk.see_fitok(); }
  const std::vector<unsigned short> &simhit_blade() { return trk.simhit_blade(); }
  const std::vector<unsigned short> &inv_subdet() { return trk.inv_subdet(); }
  const std::vector<unsigned short> &pix_blade() { return trk.pix_blade(); }
  const std::vector<float> &pix_xx() { return trk.pix_xx(); }
  const std::vector<float> &pix_xy() { return trk.pix_xy(); }
  const std::vector<unsigned short> &simhit_panel() { return trk.simhit_panel(); }
  const std::vector<float> &sim_pz() { return trk.sim_pz(); }
  const std::vector<float> &trk_dxy() { return trk.trk_dxy(); }
  const std::vector<float> &sim_px() { return trk.sim_px(); }
  const std::vector<float> &trk_lambda() { return trk.trk_lambda(); }
  const std::vector<float> &see_stateCcov12() { return trk.see_stateCcov12(); }
  const std::vector<float> &sim_pt() { return trk.sim_pt(); }
  const std::vector<float> &sim_py() { return trk.sim_py(); }
  const std::vector<std::vector<int> > &sim_decayVtxIdx() { return trk.sim_decayVtxIdx(); }
  const std::vector<unsigned int> &pix_detId() { return trk.pix_detId(); }
  const std::vector<float> &trk_eta() { return trk.trk_eta(); }
  const std::vector<float> &see_dxy() { return trk.see_dxy(); }
  const std::vector<int> &sim_isFromBHadron() { return trk.sim_isFromBHadron(); }
  const std::vector<float> &simhit_eloss() { return trk.simhit_eloss(); }
  const std::vector<float> &see_stateCcov11() { return trk.see_stateCcov11(); }
  const std::vector<float> &simhit_pz() { return trk.simhit_pz(); }
  const std::vector<int> &sim_pdgId() { return trk.sim_pdgId(); }
  const std::vector<unsigned short> &trk_stopReason() { return trk.trk_stopReason(); }
  const std::vector<float> &sim_pca_phi() { return trk.sim_pca_phi(); }
  const std::vector<unsigned short> &simhit_isLower() { return trk.simhit_isLower(); }
  const std::vector<unsigned short> &inv_ring() { return trk.inv_ring(); }
  const std::vector<std::vector<int> > &ph2_simHitIdx() { return trk.ph2_simHitIdx(); }
  const std::vector<unsigned short> &simhit_order() { return trk.simhit_order(); }
  const std::vector<float> &trk_dxyClosestPV() { return trk.trk_dxyClosestPV(); }
  const std::vector<float> &pix_z() { return trk.pix_z(); }
  const std::vector<float> &pix_y() { return trk.pix_y(); }
  const std::vector<float> &pix_x() { return trk.pix_x(); }
  const std::vector<std::vector<int> > &see_hitType() { return trk.see_hitType(); }
  const std::vector<float> &see_statePt() { return trk.see_statePt(); }
  const std::vector<std::vector<int> > &simvtx_sourceSimIdx() { return trk.simvtx_sourceSimIdx(); }
  const unsigned long long &event() { return trk.event(); }
  const std::vector<unsigned short> &pix_module() { return trk.pix_module(); }
  const std::vector<unsigned short> &ph2_side() { return trk.ph2_side(); }
  const std::vector<float> &trk_bestSimTrkNChi2() { return trk.trk_bestSimTrkNChi2(); }
  const std::vector<float> &see_stateTrajPy() { return trk.see_stateTrajPy(); }
  const std::vector<unsigned short> &inv_type() { return trk.inv_type(); }
  const float &bsp_z() { return trk.bsp_z(); }
  const float &bsp_y() { return trk.bsp_y(); }
  const std::vector<float> &simhit_py() { return trk.simhit_py(); }
  const std::vector<std::vector<int> > &see_simTrkIdx() { return trk.see_simTrkIdx(); }
  const std::vector<float> &see_stateTrajGlbZ() { return trk.see_stateTrajGlbZ(); }
  const std::vector<float> &see_stateTrajGlbX() { return trk.see_stateTrajGlbX(); }
  const std::vector<float> &see_stateTrajGlbY() { return trk.see_stateTrajGlbY(); }
  const std::vector<unsigned int> &trk_originalAlgo() { return trk.trk_originalAlgo(); }
  const std::vector<unsigned int> &trk_nPixel() { return trk.trk_nPixel(); }
  const std::vector<float> &see_stateCcov14() { return trk.see_stateCcov14(); }
  const std::vector<float> &see_stateCcov15() { return trk.see_stateCcov15(); }
  const std::vector<float> &trk_phiErr() { return trk.trk_phiErr(); }
  const std::vector<float> &see_stateCcov13() { return trk.see_stateCcov13(); }
  const std::vector<std::vector<float> > &pix_chargeFraction() { return trk.pix_chargeFraction(); }
  const std::vector<int> &trk_q() { return trk.trk_q(); }
  const std::vector<std::vector<int> > &sim_seedIdx() { return trk.sim_seedIdx(); }
  const std::vector<float> &see_dzErr() { return trk.see_dzErr(); }
  const std::vector<unsigned int> &sim_nRecoClusters() { return trk.sim_nRecoClusters(); }
  const unsigned int &run() { return trk.run(); }
  const std::vector<std::vector<float> > &ph2_xySignificance() { return trk.ph2_xySignificance(); }
  const std::vector<float> &trk_nChi2() { return trk.trk_nChi2(); }
  const std::vector<unsigned short> &pix_layer() { return trk.pix_layer(); }
  const std::vector<std::vector<float> > &pix_xySignificance() { return trk.pix_xySignificance(); }
  const std::vector<float> &sim_pca_eta() { return trk.sim_pca_eta(); }
  const std::vector<float> &see_bestSimTrkShareFrac() { return trk.see_bestSimTrkShareFrac(); }
  const std::vector<float> &see_etaErr() { return trk.see_etaErr(); }
  const std::vector<float> &trk_bestSimTrkShareFracSimDenom() { return trk.trk_bestSimTrkShareFracSimDenom(); }
  const float &bsp_sigmaz() { return trk.bsp_sigmaz(); }
  const float &bsp_sigmay() { return trk.bsp_sigmay(); }
  const float &bsp_sigmax() { return trk.bsp_sigmax(); }
  const std::vector<unsigned short> &pix_ladder() { return trk.pix_ladder(); }
  const std::vector<unsigned short> &trk_qualityMask() { return trk.trk_qualityMask(); }
  const std::vector<float> &trk_ndof() { return trk.trk_ndof(); }
  const std::vector<unsigned short> &pix_subdet() { return trk.pix_subdet(); }
  const std::vector<std::vector<int> > &ph2_seeIdx() { return trk.ph2_seeIdx(); }
  const std::vector<unsigned short> &inv_isUpper() { return trk.inv_isUpper(); }
  const std::vector<float> &ph2_zx() { return trk.ph2_zx(); }
  const std::vector<std::vector<int> > &pix_trkIdx() { return trk.pix_trkIdx(); }
  const std::vector<unsigned int> &trk_nOuterLost() { return trk.trk_nOuterLost(); }
  const std::vector<unsigned short> &inv_panel() { return trk.inv_panel(); }
  const std::vector<float> &vtx_z() { return trk.vtx_z(); }
  const std::vector<unsigned short> &simhit_layer() { return trk.simhit_layer(); }
  const std::vector<float> &vtx_y() { return trk.vtx_y(); }
  const std::vector<short> &ph2_isBarrel() { return trk.ph2_isBarrel(); }
  const std::vector<std::vector<int> > &pix_seeIdx() { return trk.pix_seeIdx(); }
  const std::vector<int> &trk_bestFromFirstHitSimTrkIdx() { return trk.trk_bestFromFirstHitSimTrkIdx(); }
  const std::vector<float> &simhit_px() { return trk.simhit_px(); }
  const std::vector<float> &see_stateTrajX() { return trk.see_stateTrajX(); }
  const std::vector<float> &see_stateTrajY() { return trk.see_stateTrajY(); }
  const std::vector<unsigned int> &trk_nOuterInactive() { return trk.trk_nOuterInactive(); }
  const std::vector<float> &sim_pca_dxy() { return trk.sim_pca_dxy(); }
  const std::vector<unsigned int> &trk_algo() { return trk.trk_algo(); }
  const std::vector<std::vector<int> > &trk_hitType() { return trk.trk_hitType(); }
  const std::vector<float> &trk_bestFromFirstHitSimTrkShareFrac() { return trk.trk_bestFromFirstHitSimTrkShareFrac(); }
  const std::vector<short> &inv_isBarrel() { return trk.inv_isBarrel(); }
  const std::vector<int> &simvtx_event() { return trk.simvtx_event(); }
  const std::vector<float> &ph2_z() { return trk.ph2_z(); }
  const std::vector<float> &ph2_x() { return trk.ph2_x(); }
  const std::vector<float> &ph2_y() { return trk.ph2_y(); }
  const std::vector<std::vector<int> > &sim_genPdgIds() { return trk.sim_genPdgIds(); }
  const std::vector<float> &trk_mva() { return trk.trk_mva(); }
  const std::vector<float> &see_stateCcov24() { return trk.see_stateCcov24(); }
  const std::vector<float> &trk_dzClosestPV() { return trk.trk_dzClosestPV(); }
  const std::vector<unsigned int> &see_nCluster() { return trk.see_nCluster(); }
  const std::vector<unsigned short> &inv_rod() { return trk.inv_rod(); }
  const std::vector<std::vector<int> > &trk_hitIdx() { return trk.trk_hitIdx(); }
  const std::vector<float> &see_stateCcov22() { return trk.see_stateCcov22(); }
  const std::vector<unsigned short> &pix_simType() { return trk.pix_simType(); }
  const std::vector<unsigned short> &simhit_ring() { return trk.simhit_ring(); }
  const std::vector<float> &trk_outer_px() { return trk.trk_outer_px(); }
  const std::vector<float> &trk_outer_py() { return trk.trk_outer_py(); }
  const std::vector<float> &trk_outer_pz() { return trk.trk_outer_pz(); }
  const std::vector<float> &ph2_zz() { return trk.ph2_zz(); }
  const std::vector<float> &trk_outer_pt() { return trk.trk_outer_pt(); }
  const std::vector<unsigned int> &trk_n3DLay() { return trk.trk_n3DLay(); }
  const std::vector<unsigned int> &trk_nValid() { return trk.trk_nValid(); }
  const std::vector<float> &see_ptErr() { return trk.see_ptErr(); }
  const std::vector<float> &see_stateTrajGlbPx() { return trk.see_stateTrajGlbPx(); }
  const std::vector<unsigned short> &ph2_simType() { return trk.ph2_simType(); }
  const std::vector<float> &trk_bestFromFirstHitSimTrkShareFracSimClusterDenom() {
    return trk.trk_bestFromFirstHitSimTrkShareFracSimClusterDenom();
  }
  const std::vector<float> &simvtx_x() { return trk.simvtx_x(); }
  const std::vector<float> &trk_pz() { return trk.trk_pz(); }
  const std::vector<float> &see_bestFromFirstHitSimTrkShareFrac() { return trk.see_bestFromFirstHitSimTrkShareFrac(); }
  const std::vector<float> &trk_px() { return trk.trk_px(); }
  const std::vector<float> &trk_py() { return trk.trk_py(); }
  const std::vector<int> &trk_vtxIdx() { return trk.trk_vtxIdx(); }
  const std::vector<unsigned int> &sim_nPixel() { return trk.sim_nPixel(); }
  const std::vector<float> &vtx_chi2() { return trk.vtx_chi2(); }
  const std::vector<unsigned short> &ph2_ring() { return trk.ph2_ring(); }
  const std::vector<float> &trk_pt() { return trk.trk_pt(); }
  const std::vector<float> &see_stateCcov44() { return trk.see_stateCcov44(); }
  const std::vector<float> &ph2_radL() { return trk.ph2_radL(); }
  const std::vector<float> &vtx_zErr() { return trk.vtx_zErr(); }
  const std::vector<float> &see_px() { return trk.see_px(); }
  const std::vector<float> &see_pz() { return trk.see_pz(); }
  const std::vector<float> &see_eta() { return trk.see_eta(); }
  const std::vector<int> &simvtx_bunchCrossing() { return trk.simvtx_bunchCrossing(); }
  const std::vector<float> &sim_pca_dz() { return trk.sim_pca_dz(); }
  const std::vector<float> &simvtx_y() { return trk.simvtx_y(); }
  const std::vector<unsigned short> &inv_isStack() { return trk.inv_isStack(); }
  const std::vector<unsigned int> &trk_nStrip() { return trk.trk_nStrip(); }
  const std::vector<float> &trk_etaErr() { return trk.trk_etaErr(); }
  const std::vector<std::vector<float> > &trk_simTrkNChi2() { return trk.trk_simTrkNChi2(); }
  const std::vector<float> &pix_zz() { return trk.pix_zz(); }
  const std::vector<int> &simhit_particle() { return trk.simhit_particle(); }
  const std::vector<float> &see_dz() { return trk.see_dz(); }
  const std::vector<float> &see_stateTrajPz() { return trk.see_stateTrajPz(); }
  const std::vector<float> &trk_bestSimTrkShareFrac() { return trk.trk_bestSimTrkShareFrac(); }
  const std::vector<float> &trk_lambdaErr() { return trk.trk_lambdaErr(); }
  const std::vector<std::vector<float> > &see_simTrkShareFrac() { return trk.see_simTrkShareFrac(); }
  const std::vector<std::vector<int> > &pix_simHitIdx() { return trk.pix_simHitIdx(); }
  const std::vector<std::vector<int> > &vtx_trkIdx() { return trk.vtx_trkIdx(); }
  const std::vector<unsigned short> &ph2_rod() { return trk.ph2_rod(); }
  const std::vector<float> &vtx_ndof() { return trk.vtx_ndof(); }
  const std::vector<unsigned int> &see_nPixel() { return trk.see_nPixel(); }
  const std::vector<unsigned int> &sim_nStrip() { return trk.sim_nStrip(); }
  const std::vector<int> &sim_bunchCrossing() { return trk.sim_bunchCrossing(); }
  const std::vector<float> &see_stateCcov45() { return trk.see_stateCcov45(); }
  const std::vector<unsigned short> &ph2_isStack() { return trk.ph2_isStack(); }
  const std::vector<std::vector<float> > &sim_trkShareFrac() { return trk.sim_trkShareFrac(); }
  const std::vector<std::vector<float> > &trk_simTrkShareFrac() { return trk.trk_simTrkShareFrac(); }
  const std::vector<float> &sim_phi() { return trk.sim_phi(); }
  const std::vector<unsigned short> &inv_side() { return trk.inv_side(); }
  const std::vector<short> &vtx_fake() { return trk.vtx_fake(); }
  const std::vector<unsigned int> &trk_nInactive() { return trk.trk_nInactive(); }
  const std::vector<unsigned int> &trk_nPixelLay() { return trk.trk_nPixelLay(); }
  const std::vector<float> &ph2_bbxi() { return trk.ph2_bbxi(); }
  const std::vector<float> &vtx_xErr() { return trk.vtx_xErr(); }
  const std::vector<float> &see_stateCcov25() { return trk.see_stateCcov25(); }
  const std::vector<int> &sim_parentVtxIdx() { return trk.sim_parentVtxIdx(); }
  const std::vector<float> &see_stateCcov23() { return trk.see_stateCcov23(); }
  const std::vector<ULong64_t> &trk_algoMask() { return trk.trk_algoMask(); }
  const std::vector<std::vector<int> > &trk_simTrkIdx() { return trk.trk_simTrkIdx(); }
  const std::vector<float> &see_phiErr() { return trk.see_phiErr(); }
  const std::vector<float> &trk_cotTheta() { return trk.trk_cotTheta(); }
  const std::vector<unsigned int> &see_algo() { return trk.see_algo(); }
  const std::vector<unsigned short> &simhit_module() { return trk.simhit_module(); }
  const std::vector<std::vector<int> > &simvtx_daughterSimIdx() { return trk.simvtx_daughterSimIdx(); }
  const std::vector<float> &vtx_x() { return trk.vtx_x(); }
  const std::vector<int> &trk_seedIdx() { return trk.trk_seedIdx(); }
  const std::vector<float> &simhit_y() { return trk.simhit_y(); }
  const std::vector<unsigned short> &inv_layer() { return trk.inv_layer(); }
  const std::vector<unsigned int> &trk_nLostLay() { return trk.trk_nLostLay(); }
  const std::vector<unsigned short> &ph2_isLower() { return trk.ph2_isLower(); }
  const std::vector<unsigned short> &pix_side() { return trk.pix_side(); }
  const std::vector<unsigned short> &inv_isLower() { return trk.inv_isLower(); }
  const std::vector<std::vector<int> > &ph2_trkIdx() { return trk.ph2_trkIdx(); }
  const std::vector<unsigned int> &sim_nValid() { return trk.sim_nValid(); }
  const std::vector<int> &simhit_simTrkIdx() { return trk.simhit_simTrkIdx(); }
  const std::vector<unsigned short> &see_nCands() { return trk.see_nCands(); }
  const std::vector<int> &see_bestSimTrkIdx() { return trk.see_bestSimTrkIdx(); }
  const std::vector<float> &vtx_yErr() { return trk.vtx_yErr(); }
  const std::vector<float> &trk_dzPV() { return trk.trk_dzPV(); }
  const std::vector<float> &ph2_xy() { return trk.ph2_xy(); }
  const std::vector<unsigned short> &inv_module() { return trk.inv_module(); }
  const std::vector<float> &see_stateCcov55() { return trk.see_stateCcov55(); }
  const std::vector<unsigned short> &pix_panel() { return trk.pix_panel(); }
  const std::vector<unsigned short> &inv_ladder() { return trk.inv_ladder(); }
  const std::vector<float> &ph2_xx() { return trk.ph2_xx(); }
  const std::vector<float> &sim_pca_cotTheta() { return trk.sim_pca_cotTheta(); }
  const std::vector<int> &simpv_idx() { return trk.simpv_idx(); }
  const std::vector<float> &trk_inner_pz() { return trk.trk_inner_pz(); }
  const std::vector<float> &see_chi2() { return trk.see_chi2(); }
  const std::vector<float> &see_stateCcov35() { return trk.see_stateCcov35(); }
  const std::vector<float> &see_stateCcov33() { return trk.see_stateCcov33(); }
  const std::vector<unsigned int> &inv_detId() { return trk.inv_detId(); }
  const std::vector<unsigned int> &see_offset() { return trk.see_offset(); }
  const std::vector<unsigned int> &sim_nLay() { return trk.sim_nLay(); }
  const std::vector<std::vector<int> > &sim_simHitIdx() { return trk.sim_simHitIdx(); }
  const std::vector<unsigned short> &simhit_isUpper() { return trk.simhit_isUpper(); }
  const std::vector<float> &see_stateCcov00() { return trk.see_stateCcov00(); }
  const std::vector<unsigned short> &see_stopReason() { return trk.see_stopReason(); }
  const std::vector<short> &vtx_valid() { return trk.vtx_valid(); }
  const unsigned int &lumi() { return trk.lumi(); }
  const std::vector<float> &trk_refpoint_x() { return trk.trk_refpoint_x(); }
  const std::vector<float> &trk_refpoint_y() { return trk.trk_refpoint_y(); }
  const std::vector<float> &trk_refpoint_z() { return trk.trk_refpoint_z(); }
  const std::vector<unsigned int> &sim_n3DLay() { return trk.sim_n3DLay(); }
  const std::vector<unsigned int> &see_nPhase2OT() { return trk.see_nPhase2OT(); }
  const std::vector<float> &trk_bestFromFirstHitSimTrkShareFracSimDenom() {
    return trk.trk_bestFromFirstHitSimTrkShareFracSimDenom();
  }
  const std::vector<float> &ph2_yy() { return trk.ph2_yy(); }
  const std::vector<float> &ph2_yz() { return trk.ph2_yz(); }
  const std::vector<unsigned short> &inv_blade() { return trk.inv_blade(); }
  const std::vector<float> &trk_ptErr() { return trk.trk_ptErr(); }
  const std::vector<float> &pix_zx() { return trk.pix_zx(); }
  const std::vector<float> &simvtx_z() { return trk.simvtx_z(); }
  const std::vector<unsigned int> &sim_nTrackerHits() { return trk.sim_nTrackerHits(); }
  const std::vector<unsigned short> &ph2_subdet() { return trk.ph2_subdet(); }
  const std::vector<float> &see_stateTrajPx() { return trk.see_stateTrajPx(); }
  const std::vector<std::vector<int> > &simhit_hitIdx() { return trk.simhit_hitIdx(); }
  const std::vector<unsigned short> &simhit_ladder() { return trk.simhit_ladder(); }
  const std::vector<unsigned short> &ph2_layer() { return trk.ph2_layer(); }
  const std::vector<float> &see_phi() { return trk.see_phi(); }
  const std::vector<float> &trk_nChi2_1Dmod() { return trk.trk_nChi2_1Dmod(); }
  const std::vector<float> &trk_inner_py() { return trk.trk_inner_py(); }
  const std::vector<float> &trk_inner_px() { return trk.trk_inner_px(); }
  const std::vector<float> &trk_dxyErr() { return trk.trk_dxyErr(); }
  const std::vector<unsigned int> &sim_nPixelLay() { return trk.sim_nPixelLay(); }
  const std::vector<unsigned int> &see_nValid() { return trk.see_nValid(); }
  const std::vector<float> &trk_inner_pt() { return trk.trk_inner_pt(); }
  const std::vector<float> &see_stateTrajGlbPy() { return trk.see_stateTrajGlbPy(); }
}  // namespace tas
