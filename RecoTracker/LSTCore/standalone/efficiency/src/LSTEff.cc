#include "LSTEff.h"
LSTEff lstEff;

void LSTEff::Init(TTree *tree) {
  tree->SetMakeClass(1);
  pT5_occupancies_branch = 0;
  if (tree->GetBranch("pT5_occupancies") != 0) {
    pT5_occupancies_branch = tree->GetBranch("pT5_occupancies");
    if (pT5_occupancies_branch) {
      pT5_occupancies_branch->SetAddress(&pT5_occupancies_);
    }
  }
  t3_phi_branch = 0;
  if (tree->GetBranch("t3_phi") != 0) {
    t3_phi_branch = tree->GetBranch("t3_phi");
    if (t3_phi_branch) {
      t3_phi_branch->SetAddress(&t3_phi_);
    }
  }
  t5_score_rphisum_branch = 0;
  if (tree->GetBranch("t5_score_rphisum") != 0) {
    t5_score_rphisum_branch = tree->GetBranch("t5_score_rphisum");
    if (t5_score_rphisum_branch) {
      t5_score_rphisum_branch->SetAddress(&t5_score_rphisum_);
    }
  }
  pT4_isFake_branch = 0;
  if (tree->GetBranch("pT4_isFake") != 0) {
    pT4_isFake_branch = tree->GetBranch("pT4_isFake");
    if (pT4_isFake_branch) {
      pT4_isFake_branch->SetAddress(&pT4_isFake_);
    }
  }
  t3_isDuplicate_branch = 0;
  if (tree->GetBranch("t3_isDuplicate") != 0) {
    t3_isDuplicate_branch = tree->GetBranch("t3_isDuplicate");
    if (t3_isDuplicate_branch) {
      t3_isDuplicate_branch->SetAddress(&t3_isDuplicate_);
    }
  }
  sim_event_branch = 0;
  if (tree->GetBranch("sim_event") != 0) {
    sim_event_branch = tree->GetBranch("sim_event");
    if (sim_event_branch) {
      sim_event_branch->SetAddress(&sim_event_);
    }
  }
  sim_q_branch = 0;
  if (tree->GetBranch("sim_q") != 0) {
    sim_q_branch = tree->GetBranch("sim_q");
    if (sim_q_branch) {
      sim_q_branch->SetAddress(&sim_q_);
    }
  }
  sim_eta_branch = 0;
  if (tree->GetBranch("sim_eta") != 0) {
    sim_eta_branch = tree->GetBranch("sim_eta");
    if (sim_eta_branch) {
      sim_eta_branch->SetAddress(&sim_eta_);
    }
  }
  pT3_foundDuplicate_branch = 0;
  if (tree->GetBranch("pT3_foundDuplicate") != 0) {
    pT3_foundDuplicate_branch = tree->GetBranch("pT3_foundDuplicate");
    if (pT3_foundDuplicate_branch) {
      pT3_foundDuplicate_branch->SetAddress(&pT3_foundDuplicate_);
    }
  }
  sim_len_branch = 0;
  if (tree->GetBranch("sim_len") != 0) {
    sim_len_branch = tree->GetBranch("sim_len");
    if (sim_len_branch) {
      sim_len_branch->SetAddress(&sim_len_);
    }
  }
  pureTCE_isDuplicate_branch = 0;
  if (tree->GetBranch("pureTCE_isDuplicate") != 0) {
    pureTCE_isDuplicate_branch = tree->GetBranch("pureTCE_isDuplicate");
    if (pureTCE_isDuplicate_branch) {
      pureTCE_isDuplicate_branch->SetAddress(&pureTCE_isDuplicate_);
    }
  }
  pT3_score_branch = 0;
  if (tree->GetBranch("pT3_score") != 0) {
    pT3_score_branch = tree->GetBranch("pT3_score");
    if (pT3_score_branch) {
      pT3_score_branch->SetAddress(&pT3_score_);
    }
  }
  t5_eta_branch = 0;
  if (tree->GetBranch("t5_eta") != 0) {
    t5_eta_branch = tree->GetBranch("t5_eta");
    if (t5_eta_branch) {
      t5_eta_branch->SetAddress(&t5_eta_);
    }
  }
  sim_denom_branch = 0;
  if (tree->GetBranch("sim_denom") != 0) {
    sim_denom_branch = tree->GetBranch("sim_denom");
    if (sim_denom_branch) {
      sim_denom_branch->SetAddress(&sim_denom_);
    }
  }
  pT5_isDuplicate_branch = 0;
  if (tree->GetBranch("pT5_isDuplicate") != 0) {
    pT5_isDuplicate_branch = tree->GetBranch("pT5_isDuplicate");
    if (pT5_isDuplicate_branch) {
      pT5_isDuplicate_branch->SetAddress(&pT5_isDuplicate_);
    }
  }
  sim_tce_matched_branch = 0;
  if (tree->GetBranch("sim_tce_matched") != 0) {
    sim_tce_matched_branch = tree->GetBranch("sim_tce_matched");
    if (sim_tce_matched_branch) {
      sim_tce_matched_branch->SetAddress(&sim_tce_matched_);
    }
  }
  pT3_isDuplicate_branch = 0;
  if (tree->GetBranch("pT3_isDuplicate") != 0) {
    pT3_isDuplicate_branch = tree->GetBranch("pT3_isDuplicate");
    if (pT3_isDuplicate_branch) {
      pT3_isDuplicate_branch->SetAddress(&pT3_isDuplicate_);
    }
  }
  tc_isDuplicate_branch = 0;
  if (tree->GetBranch("tc_isDuplicate") != 0) {
    tc_isDuplicate_branch = tree->GetBranch("tc_isDuplicate");
    if (tc_isDuplicate_branch) {
      tc_isDuplicate_branch->SetAddress(&tc_isDuplicate_);
    }
  }
  pT3_eta_2_branch = 0;
  if (tree->GetBranch("pT3_eta_2") != 0) {
    pT3_eta_2_branch = tree->GetBranch("pT3_eta_2");
    if (pT3_eta_2_branch) {
      pT3_eta_2_branch->SetAddress(&pT3_eta_2_);
    }
  }
  sim_pT3_matched_branch = 0;
  if (tree->GetBranch("sim_pT3_matched") != 0) {
    sim_pT3_matched_branch = tree->GetBranch("sim_pT3_matched");
    if (sim_pT3_matched_branch) {
      sim_pT3_matched_branch->SetAddress(&sim_pT3_matched_);
    }
  }
  pureTCE_rzChiSquared_branch = 0;
  if (tree->GetBranch("pureTCE_rzChiSquared") != 0) {
    pureTCE_rzChiSquared_branch = tree->GetBranch("pureTCE_rzChiSquared");
    if (pureTCE_rzChiSquared_branch) {
      pureTCE_rzChiSquared_branch->SetAddress(&pureTCE_rzChiSquared_);
    }
  }
  t4_isDuplicate_branch = 0;
  if (tree->GetBranch("t4_isDuplicate") != 0) {
    t4_isDuplicate_branch = tree->GetBranch("t4_isDuplicate");
    if (t4_isDuplicate_branch) {
      t4_isDuplicate_branch->SetAddress(&t4_isDuplicate_);
    }
  }
  pureTCE_eta_branch = 0;
  if (tree->GetBranch("pureTCE_eta") != 0) {
    pureTCE_eta_branch = tree->GetBranch("pureTCE_eta");
    if (pureTCE_eta_branch) {
      pureTCE_eta_branch->SetAddress(&pureTCE_eta_);
    }
  }
  tce_rPhiChiSquared_branch = 0;
  if (tree->GetBranch("tce_rPhiChiSquared") != 0) {
    tce_rPhiChiSquared_branch = tree->GetBranch("tce_rPhiChiSquared");
    if (tce_rPhiChiSquared_branch) {
      tce_rPhiChiSquared_branch->SetAddress(&tce_rPhiChiSquared_);
    }
  }
  pureTCE_anchorType_branch = 0;
  if (tree->GetBranch("pureTCE_anchorType") != 0) {
    pureTCE_anchorType_branch = tree->GetBranch("pureTCE_anchorType");
    if (pureTCE_anchorType_branch) {
      pureTCE_anchorType_branch->SetAddress(&pureTCE_anchorType_);
    }
  }
  pureTCE_pt_branch = 0;
  if (tree->GetBranch("pureTCE_pt") != 0) {
    pureTCE_pt_branch = tree->GetBranch("pureTCE_pt");
    if (pureTCE_pt_branch) {
      pureTCE_pt_branch->SetAddress(&pureTCE_pt_);
    }
  }
  sim_pt_branch = 0;
  if (tree->GetBranch("sim_pt") != 0) {
    sim_pt_branch = tree->GetBranch("sim_pt");
    if (sim_pt_branch) {
      sim_pt_branch->SetAddress(&sim_pt_);
    }
  }
  t5_eta_2_branch = 0;
  if (tree->GetBranch("t5_eta_2") != 0) {
    t5_eta_2_branch = tree->GetBranch("t5_eta_2");
    if (t5_eta_2_branch) {
      t5_eta_2_branch->SetAddress(&t5_eta_2_);
    }
  }
  pLS_eta_branch = 0;
  if (tree->GetBranch("pLS_eta") != 0) {
    pLS_eta_branch = tree->GetBranch("pLS_eta");
    if (pLS_eta_branch) {
      pLS_eta_branch->SetAddress(&pLS_eta_);
    }
  }
  sim_pdgId_branch = 0;
  if (tree->GetBranch("sim_pdgId") != 0) {
    sim_pdgId_branch = tree->GetBranch("sim_pdgId");
    if (sim_pdgId_branch) {
      sim_pdgId_branch->SetAddress(&sim_pdgId_);
    }
  }
  t3_eta_branch = 0;
  if (tree->GetBranch("t3_eta") != 0) {
    t3_eta_branch = tree->GetBranch("t3_eta");
    if (t3_eta_branch) {
      t3_eta_branch->SetAddress(&t3_eta_);
    }
  }
  tce_layer_binary_branch = 0;
  if (tree->GetBranch("tce_layer_binary") != 0) {
    tce_layer_binary_branch = tree->GetBranch("tce_layer_binary");
    if (tce_layer_binary_branch) {
      tce_layer_binary_branch->SetAddress(&tce_layer_binary_);
    }
  }
  sim_TC_matched_nonextended_branch = 0;
  if (tree->GetBranch("sim_TC_matched_nonextended") != 0) {
    sim_TC_matched_nonextended_branch = tree->GetBranch("sim_TC_matched_nonextended");
    if (sim_TC_matched_nonextended_branch) {
      sim_TC_matched_nonextended_branch->SetAddress(&sim_TC_matched_nonextended_);
    }
  }
  t4_occupancies_branch = 0;
  if (tree->GetBranch("t4_occupancies") != 0) {
    t4_occupancies_branch = tree->GetBranch("t4_occupancies");
    if (t4_occupancies_branch) {
      t4_occupancies_branch->SetAddress(&t4_occupancies_);
    }
  }
  tce_eta_branch = 0;
  if (tree->GetBranch("tce_eta") != 0) {
    tce_eta_branch = tree->GetBranch("tce_eta");
    if (tce_eta_branch) {
      tce_eta_branch->SetAddress(&tce_eta_);
    }
  }
  tce_isDuplicate_branch = 0;
  if (tree->GetBranch("tce_isDuplicate") != 0) {
    tce_isDuplicate_branch = tree->GetBranch("tce_isDuplicate");
    if (tce_isDuplicate_branch) {
      tce_isDuplicate_branch->SetAddress(&tce_isDuplicate_);
    }
  }
  pT5_matched_simIdx_branch = 0;
  if (tree->GetBranch("pT5_matched_simIdx") != 0) {
    pT5_matched_simIdx_branch = tree->GetBranch("pT5_matched_simIdx");
    if (pT5_matched_simIdx_branch) {
      pT5_matched_simIdx_branch->SetAddress(&pT5_matched_simIdx_);
    }
  }
  sim_tcIdx_branch = 0;
  if (tree->GetBranch("sim_tcIdx") != 0) {
    sim_tcIdx_branch = tree->GetBranch("sim_tcIdx");
    if (sim_tcIdx_branch) {
      sim_tcIdx_branch->SetAddress(&sim_tcIdx_);
    }
  }
  t5_phi_2_branch = 0;
  if (tree->GetBranch("t5_phi_2") != 0) {
    t5_phi_2_branch = tree->GetBranch("t5_phi_2");
    if (t5_phi_2_branch) {
      t5_phi_2_branch->SetAddress(&t5_phi_2_);
    }
  }
  pureTCE_maxHitMatchedCounts_branch = 0;
  if (tree->GetBranch("pureTCE_maxHitMatchedCounts") != 0) {
    pureTCE_maxHitMatchedCounts_branch = tree->GetBranch("pureTCE_maxHitMatchedCounts");
    if (pureTCE_maxHitMatchedCounts_branch) {
      pureTCE_maxHitMatchedCounts_branch->SetAddress(&pureTCE_maxHitMatchedCounts_);
    }
  }
  t5_matched_simIdx_branch = 0;
  if (tree->GetBranch("t5_matched_simIdx") != 0) {
    t5_matched_simIdx_branch = tree->GetBranch("t5_matched_simIdx");
    if (t5_matched_simIdx_branch) {
      t5_matched_simIdx_branch->SetAddress(&t5_matched_simIdx_);
    }
  }
  module_subdets_branch = 0;
  if (tree->GetBranch("module_subdets") != 0) {
    module_subdets_branch = tree->GetBranch("module_subdets");
    if (module_subdets_branch) {
      module_subdets_branch->SetAddress(&module_subdets_);
    }
  }
  tce_anchorType_branch = 0;
  if (tree->GetBranch("tce_anchorType") != 0) {
    tce_anchorType_branch = tree->GetBranch("tce_anchorType");
    if (tce_anchorType_branch) {
      tce_anchorType_branch->SetAddress(&tce_anchorType_);
    }
  }
  tce_nHitOverlaps_branch = 0;
  if (tree->GetBranch("tce_nHitOverlaps") != 0) {
    tce_nHitOverlaps_branch = tree->GetBranch("tce_nHitOverlaps");
    if (tce_nHitOverlaps_branch) {
      tce_nHitOverlaps_branch->SetAddress(&tce_nHitOverlaps_);
    }
  }
  t3_isFake_branch = 0;
  if (tree->GetBranch("t3_isFake") != 0) {
    t3_isFake_branch = tree->GetBranch("t3_isFake");
    if (t3_isFake_branch) {
      t3_isFake_branch->SetAddress(&t3_isFake_);
    }
  }
  tce_phi_branch = 0;
  if (tree->GetBranch("tce_phi") != 0) {
    tce_phi_branch = tree->GetBranch("tce_phi");
    if (tce_phi_branch) {
      tce_phi_branch->SetAddress(&tce_phi_);
    }
  }
  t5_isFake_branch = 0;
  if (tree->GetBranch("t5_isFake") != 0) {
    t5_isFake_branch = tree->GetBranch("t5_isFake");
    if (t5_isFake_branch) {
      t5_isFake_branch->SetAddress(&t5_isFake_);
    }
  }
  md_occupancies_branch = 0;
  if (tree->GetBranch("md_occupancies") != 0) {
    md_occupancies_branch = tree->GetBranch("md_occupancies");
    if (md_occupancies_branch) {
      md_occupancies_branch->SetAddress(&md_occupancies_);
    }
  }
  t5_hitIdxs_branch = 0;
  if (tree->GetBranch("t5_hitIdxs") != 0) {
    t5_hitIdxs_branch = tree->GetBranch("t5_hitIdxs");
    if (t5_hitIdxs_branch) {
      t5_hitIdxs_branch->SetAddress(&t5_hitIdxs_);
    }
  }
  sim_pT3_types_branch = 0;
  if (tree->GetBranch("sim_pT3_types") != 0) {
    sim_pT3_types_branch = tree->GetBranch("sim_pT3_types");
    if (sim_pT3_types_branch) {
      sim_pT3_types_branch->SetAddress(&sim_pT3_types_);
    }
  }
  sim_pureTCE_types_branch = 0;
  if (tree->GetBranch("sim_pureTCE_types") != 0) {
    sim_pureTCE_types_branch = tree->GetBranch("sim_pureTCE_types");
    if (sim_pureTCE_types_branch) {
      sim_pureTCE_types_branch->SetAddress(&sim_pureTCE_types_);
    }
  }
  t4_phi_branch = 0;
  if (tree->GetBranch("t4_phi") != 0) {
    t4_phi_branch = tree->GetBranch("t4_phi");
    if (t4_phi_branch) {
      t4_phi_branch->SetAddress(&t4_phi_);
    }
  }
  t5_phi_branch = 0;
  if (tree->GetBranch("t5_phi") != 0) {
    t5_phi_branch = tree->GetBranch("t5_phi");
    if (t5_phi_branch) {
      t5_phi_branch->SetAddress(&t5_phi_);
    }
  }
  pT5_hitIdxs_branch = 0;
  if (tree->GetBranch("pT5_hitIdxs") != 0) {
    pT5_hitIdxs_branch = tree->GetBranch("pT5_hitIdxs");
    if (pT5_hitIdxs_branch) {
      pT5_hitIdxs_branch->SetAddress(&pT5_hitIdxs_);
    }
  }
  t5_pt_branch = 0;
  if (tree->GetBranch("t5_pt") != 0) {
    t5_pt_branch = tree->GetBranch("t5_pt");
    if (t5_pt_branch) {
      t5_pt_branch->SetAddress(&t5_pt_);
    }
  }
  pT5_phi_branch = 0;
  if (tree->GetBranch("pT5_phi") != 0) {
    pT5_phi_branch = tree->GetBranch("pT5_phi");
    if (pT5_phi_branch) {
      pT5_phi_branch->SetAddress(&pT5_phi_);
    }
  }
  pureTCE_isFake_branch = 0;
  if (tree->GetBranch("pureTCE_isFake") != 0) {
    pureTCE_isFake_branch = tree->GetBranch("pureTCE_isFake");
    if (pureTCE_isFake_branch) {
      pureTCE_isFake_branch->SetAddress(&pureTCE_isFake_);
    }
  }
  tce_pt_branch = 0;
  if (tree->GetBranch("tce_pt") != 0) {
    tce_pt_branch = tree->GetBranch("tce_pt");
    if (tce_pt_branch) {
      tce_pt_branch->SetAddress(&tce_pt_);
    }
  }
  tc_isFake_branch = 0;
  if (tree->GetBranch("tc_isFake") != 0) {
    tc_isFake_branch = tree->GetBranch("tc_isFake");
    if (tc_isFake_branch) {
      tc_isFake_branch->SetAddress(&tc_isFake_);
    }
  }
  pT3_isFake_branch = 0;
  if (tree->GetBranch("pT3_isFake") != 0) {
    pT3_isFake_branch = tree->GetBranch("pT3_isFake");
    if (pT3_isFake_branch) {
      pT3_isFake_branch->SetAddress(&pT3_isFake_);
    }
  }
  tce_nLayerOverlaps_branch = 0;
  if (tree->GetBranch("tce_nLayerOverlaps") != 0) {
    tce_nLayerOverlaps_branch = tree->GetBranch("tce_nLayerOverlaps");
    if (tce_nLayerOverlaps_branch) {
      tce_nLayerOverlaps_branch->SetAddress(&tce_nLayerOverlaps_);
    }
  }
  tc_sim_branch = 0;
  if (tree->GetBranch("tc_sim") != 0) {
    tc_sim_branch = tree->GetBranch("tc_sim");
    if (tc_sim_branch) {
      tc_sim_branch->SetAddress(&tc_sim_);
    }
  }
  sim_pLS_types_branch = 0;
  if (tree->GetBranch("sim_pLS_types") != 0) {
    sim_pLS_types_branch = tree->GetBranch("sim_pLS_types");
    if (sim_pLS_types_branch) {
      sim_pLS_types_branch->SetAddress(&sim_pLS_types_);
    }
  }
  sim_pca_dxy_branch = 0;
  if (tree->GetBranch("sim_pca_dxy") != 0) {
    sim_pca_dxy_branch = tree->GetBranch("sim_pca_dxy");
    if (sim_pca_dxy_branch) {
      sim_pca_dxy_branch->SetAddress(&sim_pca_dxy_);
    }
  }
  pT4_phi_branch = 0;
  if (tree->GetBranch("pT4_phi") != 0) {
    pT4_phi_branch = tree->GetBranch("pT4_phi");
    if (pT4_phi_branch) {
      pT4_phi_branch->SetAddress(&pT4_phi_);
    }
  }
  sim_hits_branch = 0;
  if (tree->GetBranch("sim_hits") != 0) {
    sim_hits_branch = tree->GetBranch("sim_hits");
    if (sim_hits_branch) {
      sim_hits_branch->SetAddress(&sim_hits_);
    }
  }
  pLS_phi_branch = 0;
  if (tree->GetBranch("pLS_phi") != 0) {
    pLS_phi_branch = tree->GetBranch("pLS_phi");
    if (pLS_phi_branch) {
      pLS_phi_branch->SetAddress(&pLS_phi_);
    }
  }
  sim_pureTCE_matched_branch = 0;
  if (tree->GetBranch("sim_pureTCE_matched") != 0) {
    sim_pureTCE_matched_branch = tree->GetBranch("sim_pureTCE_matched");
    if (sim_pureTCE_matched_branch) {
      sim_pureTCE_matched_branch->SetAddress(&sim_pureTCE_matched_);
    }
  }
  t3_occupancies_branch = 0;
  if (tree->GetBranch("t3_occupancies") != 0) {
    t3_occupancies_branch = tree->GetBranch("t3_occupancies");
    if (t3_occupancies_branch) {
      t3_occupancies_branch->SetAddress(&t3_occupancies_);
    }
  }
  t5_foundDuplicate_branch = 0;
  if (tree->GetBranch("t5_foundDuplicate") != 0) {
    t5_foundDuplicate_branch = tree->GetBranch("t5_foundDuplicate");
    if (t5_foundDuplicate_branch) {
      t5_foundDuplicate_branch->SetAddress(&t5_foundDuplicate_);
    }
  }
  sim_pT4_types_branch = 0;
  if (tree->GetBranch("sim_pT4_types") != 0) {
    sim_pT4_types_branch = tree->GetBranch("sim_pT4_types");
    if (sim_pT4_types_branch) {
      sim_pT4_types_branch->SetAddress(&sim_pT4_types_);
    }
  }
  t4_isFake_branch = 0;
  if (tree->GetBranch("t4_isFake") != 0) {
    t4_isFake_branch = tree->GetBranch("t4_isFake");
    if (t4_isFake_branch) {
      t4_isFake_branch->SetAddress(&t4_isFake_);
    }
  }
  simvtx_x_branch = 0;
  if (tree->GetBranch("simvtx_x") != 0) {
    simvtx_x_branch = tree->GetBranch("simvtx_x");
    if (simvtx_x_branch) {
      simvtx_x_branch->SetAddress(&simvtx_x_);
    }
  }
  simvtx_y_branch = 0;
  if (tree->GetBranch("simvtx_y") != 0) {
    simvtx_y_branch = tree->GetBranch("simvtx_y");
    if (simvtx_y_branch) {
      simvtx_y_branch->SetAddress(&simvtx_y_);
    }
  }
  simvtx_z_branch = 0;
  if (tree->GetBranch("simvtx_z") != 0) {
    simvtx_z_branch = tree->GetBranch("simvtx_z");
    if (simvtx_z_branch) {
      simvtx_z_branch->SetAddress(&simvtx_z_);
    }
  }
  sim_T4_matched_branch = 0;
  if (tree->GetBranch("sim_T4_matched") != 0) {
    sim_T4_matched_branch = tree->GetBranch("sim_T4_matched");
    if (sim_T4_matched_branch) {
      sim_T4_matched_branch->SetAddress(&sim_T4_matched_);
    }
  }
  sim_isGood_branch = 0;
  if (tree->GetBranch("sim_isGood") != 0) {
    sim_isGood_branch = tree->GetBranch("sim_isGood");
    if (sim_isGood_branch) {
      sim_isGood_branch->SetAddress(&sim_isGood_);
    }
  }
  pT3_pt_branch = 0;
  if (tree->GetBranch("pT3_pt") != 0) {
    pT3_pt_branch = tree->GetBranch("pT3_pt");
    if (pT3_pt_branch) {
      pT3_pt_branch->SetAddress(&pT3_pt_);
    }
  }
  tc_pt_branch = 0;
  if (tree->GetBranch("tc_pt") != 0) {
    tc_pt_branch = tree->GetBranch("tc_pt");
    if (tc_pt_branch) {
      tc_pt_branch->SetAddress(&tc_pt_);
    }
  }
  pT3_phi_2_branch = 0;
  if (tree->GetBranch("pT3_phi_2") != 0) {
    pT3_phi_2_branch = tree->GetBranch("pT3_phi_2");
    if (pT3_phi_2_branch) {
      pT3_phi_2_branch->SetAddress(&pT3_phi_2_);
    }
  }
  pT5_pt_branch = 0;
  if (tree->GetBranch("pT5_pt") != 0) {
    pT5_pt_branch = tree->GetBranch("pT5_pt");
    if (pT5_pt_branch) {
      pT5_pt_branch->SetAddress(&pT5_pt_);
    }
  }
  pureTCE_rPhiChiSquared_branch = 0;
  if (tree->GetBranch("pureTCE_rPhiChiSquared") != 0) {
    pureTCE_rPhiChiSquared_branch = tree->GetBranch("pureTCE_rPhiChiSquared");
    if (pureTCE_rPhiChiSquared_branch) {
      pureTCE_rPhiChiSquared_branch->SetAddress(&pureTCE_rPhiChiSquared_);
    }
  }
  pT5_score_branch = 0;
  if (tree->GetBranch("pT5_score") != 0) {
    pT5_score_branch = tree->GetBranch("pT5_score");
    if (pT5_score_branch) {
      pT5_score_branch->SetAddress(&pT5_score_);
    }
  }
  sim_phi_branch = 0;
  if (tree->GetBranch("sim_phi") != 0) {
    sim_phi_branch = tree->GetBranch("sim_phi");
    if (sim_phi_branch) {
      sim_phi_branch->SetAddress(&sim_phi_);
    }
  }
  pT5_isFake_branch = 0;
  if (tree->GetBranch("pT5_isFake") != 0) {
    pT5_isFake_branch = tree->GetBranch("pT5_isFake");
    if (pT5_isFake_branch) {
      pT5_isFake_branch->SetAddress(&pT5_isFake_);
    }
  }
  tc_maxHitMatchedCounts_branch = 0;
  if (tree->GetBranch("tc_maxHitMatchedCounts") != 0) {
    tc_maxHitMatchedCounts_branch = tree->GetBranch("tc_maxHitMatchedCounts");
    if (tc_maxHitMatchedCounts_branch) {
      tc_maxHitMatchedCounts_branch->SetAddress(&tc_maxHitMatchedCounts_);
    }
  }
  pureTCE_nLayerOverlaps_branch = 0;
  if (tree->GetBranch("pureTCE_nLayerOverlaps") != 0) {
    pureTCE_nLayerOverlaps_branch = tree->GetBranch("pureTCE_nLayerOverlaps");
    if (pureTCE_nLayerOverlaps_branch) {
      pureTCE_nLayerOverlaps_branch->SetAddress(&pureTCE_nLayerOverlaps_);
    }
  }
  sim_pca_dz_branch = 0;
  if (tree->GetBranch("sim_pca_dz") != 0) {
    sim_pca_dz_branch = tree->GetBranch("sim_pca_dz");
    if (sim_pca_dz_branch) {
      sim_pca_dz_branch->SetAddress(&sim_pca_dz_);
    }
  }
  pureTCE_hitIdxs_branch = 0;
  if (tree->GetBranch("pureTCE_hitIdxs") != 0) {
    pureTCE_hitIdxs_branch = tree->GetBranch("pureTCE_hitIdxs");
    if (pureTCE_hitIdxs_branch) {
      pureTCE_hitIdxs_branch->SetAddress(&pureTCE_hitIdxs_);
    }
  }
  pureTCE_nHitOverlaps_branch = 0;
  if (tree->GetBranch("pureTCE_nHitOverlaps") != 0) {
    pureTCE_nHitOverlaps_branch = tree->GetBranch("pureTCE_nHitOverlaps");
    if (pureTCE_nHitOverlaps_branch) {
      pureTCE_nHitOverlaps_branch->SetAddress(&pureTCE_nHitOverlaps_);
    }
  }
  sim_pLS_matched_branch = 0;
  if (tree->GetBranch("sim_pLS_matched") != 0) {
    sim_pLS_matched_branch = tree->GetBranch("sim_pLS_matched");
    if (sim_pLS_matched_branch) {
      sim_pLS_matched_branch->SetAddress(&sim_pLS_matched_);
    }
  }
  tc_matched_simIdx_branch = 0;
  if (tree->GetBranch("tc_matched_simIdx") != 0) {
    tc_matched_simIdx_branch = tree->GetBranch("tc_matched_simIdx");
    if (tc_matched_simIdx_branch) {
      tc_matched_simIdx_branch->SetAddress(&tc_matched_simIdx_);
    }
  }
  sim_T3_matched_branch = 0;
  if (tree->GetBranch("sim_T3_matched") != 0) {
    sim_T3_matched_branch = tree->GetBranch("sim_T3_matched");
    if (sim_T3_matched_branch) {
      sim_T3_matched_branch->SetAddress(&sim_T3_matched_);
    }
  }
  pLS_score_branch = 0;
  if (tree->GetBranch("pLS_score") != 0) {
    pLS_score_branch = tree->GetBranch("pLS_score");
    if (pLS_score_branch) {
      pLS_score_branch->SetAddress(&pLS_score_);
    }
  }
  pT3_phi_branch = 0;
  if (tree->GetBranch("pT3_phi") != 0) {
    pT3_phi_branch = tree->GetBranch("pT3_phi");
    if (pT3_phi_branch) {
      pT3_phi_branch->SetAddress(&pT3_phi_);
    }
  }
  pT5_eta_branch = 0;
  if (tree->GetBranch("pT5_eta") != 0) {
    pT5_eta_branch = tree->GetBranch("pT5_eta");
    if (pT5_eta_branch) {
      pT5_eta_branch->SetAddress(&pT5_eta_);
    }
  }
  tc_phi_branch = 0;
  if (tree->GetBranch("tc_phi") != 0) {
    tc_phi_branch = tree->GetBranch("tc_phi");
    if (tc_phi_branch) {
      tc_phi_branch->SetAddress(&tc_phi_);
    }
  }
  t4_eta_branch = 0;
  if (tree->GetBranch("t4_eta") != 0) {
    t4_eta_branch = tree->GetBranch("t4_eta");
    if (t4_eta_branch) {
      t4_eta_branch->SetAddress(&t4_eta_);
    }
  }
  pLS_isFake_branch = 0;
  if (tree->GetBranch("pLS_isFake") != 0) {
    pLS_isFake_branch = tree->GetBranch("pLS_isFake");
    if (pLS_isFake_branch) {
      pLS_isFake_branch->SetAddress(&pLS_isFake_);
    }
  }
  pureTCE_matched_simIdx_branch = 0;
  if (tree->GetBranch("pureTCE_matched_simIdx") != 0) {
    pureTCE_matched_simIdx_branch = tree->GetBranch("pureTCE_matched_simIdx");
    if (pureTCE_matched_simIdx_branch) {
      pureTCE_matched_simIdx_branch->SetAddress(&pureTCE_matched_simIdx_);
    }
  }
  sim_bunchCrossing_branch = 0;
  if (tree->GetBranch("sim_bunchCrossing") != 0) {
    sim_bunchCrossing_branch = tree->GetBranch("sim_bunchCrossing");
    if (sim_bunchCrossing_branch) {
      sim_bunchCrossing_branch->SetAddress(&sim_bunchCrossing_);
    }
  }
  tc_partOfExtension_branch = 0;
  if (tree->GetBranch("tc_partOfExtension") != 0) {
    tc_partOfExtension_branch = tree->GetBranch("tc_partOfExtension");
    if (tc_partOfExtension_branch) {
      tc_partOfExtension_branch->SetAddress(&tc_partOfExtension_);
    }
  }
  pT3_eta_branch = 0;
  if (tree->GetBranch("pT3_eta") != 0) {
    pT3_eta_branch = tree->GetBranch("pT3_eta");
    if (pT3_eta_branch) {
      pT3_eta_branch->SetAddress(&pT3_eta_);
    }
  }
  sim_parentVtxIdx_branch = 0;
  if (tree->GetBranch("sim_parentVtxIdx") != 0) {
    sim_parentVtxIdx_branch = tree->GetBranch("sim_parentVtxIdx");
    if (sim_parentVtxIdx_branch) {
      sim_parentVtxIdx_branch->SetAddress(&sim_parentVtxIdx_);
    }
  }
  pureTCE_layer_binary_branch = 0;
  if (tree->GetBranch("pureTCE_layer_binary") != 0) {
    pureTCE_layer_binary_branch = tree->GetBranch("pureTCE_layer_binary");
    if (pureTCE_layer_binary_branch) {
      pureTCE_layer_binary_branch->SetAddress(&pureTCE_layer_binary_);
    }
  }
  sim_pT4_matched_branch = 0;
  if (tree->GetBranch("sim_pT4_matched") != 0) {
    sim_pT4_matched_branch = tree->GetBranch("sim_pT4_matched");
    if (sim_pT4_matched_branch) {
      sim_pT4_matched_branch->SetAddress(&sim_pT4_matched_);
    }
  }
  tc_eta_branch = 0;
  if (tree->GetBranch("tc_eta") != 0) {
    tc_eta_branch = tree->GetBranch("tc_eta");
    if (tc_eta_branch) {
      tc_eta_branch->SetAddress(&tc_eta_);
    }
  }
  sim_lengap_branch = 0;
  if (tree->GetBranch("sim_lengap") != 0) {
    sim_lengap_branch = tree->GetBranch("sim_lengap");
    if (sim_lengap_branch) {
      sim_lengap_branch->SetAddress(&sim_lengap_);
    }
  }
  sim_T5_matched_branch = 0;
  if (tree->GetBranch("sim_T5_matched") != 0) {
    sim_T5_matched_branch = tree->GetBranch("sim_T5_matched");
    if (sim_T5_matched_branch) {
      sim_T5_matched_branch->SetAddress(&sim_T5_matched_);
    }
  }
  sim_T5_types_branch = 0;
  if (tree->GetBranch("sim_T5_types") != 0) {
    sim_T5_types_branch = tree->GetBranch("sim_T5_types");
    if (sim_T5_types_branch) {
      sim_T5_types_branch->SetAddress(&sim_T5_types_);
    }
  }
  tce_matched_simIdx_branch = 0;
  if (tree->GetBranch("tce_matched_simIdx") != 0) {
    tce_matched_simIdx_branch = tree->GetBranch("tce_matched_simIdx");
    if (tce_matched_simIdx_branch) {
      tce_matched_simIdx_branch->SetAddress(&tce_matched_simIdx_);
    }
  }
  t5_isDuplicate_branch = 0;
  if (tree->GetBranch("t5_isDuplicate") != 0) {
    t5_isDuplicate_branch = tree->GetBranch("t5_isDuplicate");
    if (t5_isDuplicate_branch) {
      t5_isDuplicate_branch->SetAddress(&t5_isDuplicate_);
    }
  }
  pT3_hitIdxs_branch = 0;
  if (tree->GetBranch("pT3_hitIdxs") != 0) {
    pT3_hitIdxs_branch = tree->GetBranch("pT3_hitIdxs");
    if (pT3_hitIdxs_branch) {
      pT3_hitIdxs_branch->SetAddress(&pT3_hitIdxs_);
    }
  }
  tc_hitIdxs_branch = 0;
  if (tree->GetBranch("tc_hitIdxs") != 0) {
    tc_hitIdxs_branch = tree->GetBranch("tc_hitIdxs");
    if (tc_hitIdxs_branch) {
      tc_hitIdxs_branch->SetAddress(&tc_hitIdxs_);
    }
  }
  pT3_occupancies_branch = 0;
  if (tree->GetBranch("pT3_occupancies") != 0) {
    pT3_occupancies_branch = tree->GetBranch("pT3_occupancies");
    if (pT3_occupancies_branch) {
      pT3_occupancies_branch->SetAddress(&pT3_occupancies_);
    }
  }
  tc_occupancies_branch = 0;
  if (tree->GetBranch("tc_occupancies") != 0) {
    tc_occupancies_branch = tree->GetBranch("tc_occupancies");
    if (tc_occupancies_branch) {
      tc_occupancies_branch->SetAddress(&tc_occupancies_);
    }
  }
  sim_TC_matched_branch = 0;
  if (tree->GetBranch("sim_TC_matched") != 0) {
    sim_TC_matched_branch = tree->GetBranch("sim_TC_matched");
    if (sim_TC_matched_branch) {
      sim_TC_matched_branch->SetAddress(&sim_TC_matched_);
    }
  }
  sim_TC_matched_mask_branch = 0;
  if (tree->GetBranch("sim_TC_matched_mask") != 0) {
    sim_TC_matched_mask_branch = tree->GetBranch("sim_TC_matched_mask");
    if (sim_TC_matched_mask_branch) {
      sim_TC_matched_mask_branch->SetAddress(&sim_TC_matched_mask_);
    }
  }
  pLS_isDuplicate_branch = 0;
  if (tree->GetBranch("pLS_isDuplicate") != 0) {
    pLS_isDuplicate_branch = tree->GetBranch("pLS_isDuplicate");
    if (pLS_isDuplicate_branch) {
      pLS_isDuplicate_branch->SetAddress(&pLS_isDuplicate_);
    }
  }
  tce_anchorIndex_branch = 0;
  if (tree->GetBranch("tce_anchorIndex") != 0) {
    tce_anchorIndex_branch = tree->GetBranch("tce_anchorIndex");
    if (tce_anchorIndex_branch) {
      tce_anchorIndex_branch->SetAddress(&tce_anchorIndex_);
    }
  }
  t5_occupancies_branch = 0;
  if (tree->GetBranch("t5_occupancies") != 0) {
    t5_occupancies_branch = tree->GetBranch("t5_occupancies");
    if (t5_occupancies_branch) {
      t5_occupancies_branch->SetAddress(&t5_occupancies_);
    }
  }
  tc_type_branch = 0;
  if (tree->GetBranch("tc_type") != 0) {
    tc_type_branch = tree->GetBranch("tc_type");
    if (tc_type_branch) {
      tc_type_branch->SetAddress(&tc_type_);
    }
  }
  tce_isFake_branch = 0;
  if (tree->GetBranch("tce_isFake") != 0) {
    tce_isFake_branch = tree->GetBranch("tce_isFake");
    if (tce_isFake_branch) {
      tce_isFake_branch->SetAddress(&tce_isFake_);
    }
  }
  pLS_pt_branch = 0;
  if (tree->GetBranch("pLS_pt") != 0) {
    pLS_pt_branch = tree->GetBranch("pLS_pt");
    if (pLS_pt_branch) {
      pLS_pt_branch->SetAddress(&pLS_pt_);
    }
  }
  pureTCE_anchorIndex_branch = 0;
  if (tree->GetBranch("pureTCE_anchorIndex") != 0) {
    pureTCE_anchorIndex_branch = tree->GetBranch("pureTCE_anchorIndex");
    if (pureTCE_anchorIndex_branch) {
      pureTCE_anchorIndex_branch->SetAddress(&pureTCE_anchorIndex_);
    }
  }
  sim_T4_types_branch = 0;
  if (tree->GetBranch("sim_T4_types") != 0) {
    sim_T4_types_branch = tree->GetBranch("sim_T4_types");
    if (sim_T4_types_branch) {
      sim_T4_types_branch->SetAddress(&sim_T4_types_);
    }
  }
  pT4_isDuplicate_branch = 0;
  if (tree->GetBranch("pT4_isDuplicate") != 0) {
    pT4_isDuplicate_branch = tree->GetBranch("pT4_isDuplicate");
    if (pT4_isDuplicate_branch) {
      pT4_isDuplicate_branch->SetAddress(&pT4_isDuplicate_);
    }
  }
  t4_pt_branch = 0;
  if (tree->GetBranch("t4_pt") != 0) {
    t4_pt_branch = tree->GetBranch("t4_pt");
    if (t4_pt_branch) {
      t4_pt_branch->SetAddress(&t4_pt_);
    }
  }
  sim_TC_types_branch = 0;
  if (tree->GetBranch("sim_TC_types") != 0) {
    sim_TC_types_branch = tree->GetBranch("sim_TC_types");
    if (sim_TC_types_branch) {
      sim_TC_types_branch->SetAddress(&sim_TC_types_);
    }
  }
  sg_occupancies_branch = 0;
  if (tree->GetBranch("sg_occupancies") != 0) {
    sg_occupancies_branch = tree->GetBranch("sg_occupancies");
    if (sg_occupancies_branch) {
      sg_occupancies_branch->SetAddress(&sg_occupancies_);
    }
  }
  pT4_pt_branch = 0;
  if (tree->GetBranch("pT4_pt") != 0) {
    pT4_pt_branch = tree->GetBranch("pT4_pt");
    if (pT4_pt_branch) {
      pT4_pt_branch->SetAddress(&pT4_pt_);
    }
  }
  pureTCE_phi_branch = 0;
  if (tree->GetBranch("pureTCE_phi") != 0) {
    pureTCE_phi_branch = tree->GetBranch("pureTCE_phi");
    if (pureTCE_phi_branch) {
      pureTCE_phi_branch->SetAddress(&pureTCE_phi_);
    }
  }
  sim_vx_branch = 0;
  if (tree->GetBranch("sim_vx") != 0) {
    sim_vx_branch = tree->GetBranch("sim_vx");
    if (sim_vx_branch) {
      sim_vx_branch->SetAddress(&sim_vx_);
    }
  }
  sim_vy_branch = 0;
  if (tree->GetBranch("sim_vy") != 0) {
    sim_vy_branch = tree->GetBranch("sim_vy");
    if (sim_vy_branch) {
      sim_vy_branch->SetAddress(&sim_vy_);
    }
  }
  sim_vz_branch = 0;
  if (tree->GetBranch("sim_vz") != 0) {
    sim_vz_branch = tree->GetBranch("sim_vz");
    if (sim_vz_branch) {
      sim_vz_branch->SetAddress(&sim_vz_);
    }
  }
  tce_maxHitMatchedCounts_branch = 0;
  if (tree->GetBranch("tce_maxHitMatchedCounts") != 0) {
    tce_maxHitMatchedCounts_branch = tree->GetBranch("tce_maxHitMatchedCounts");
    if (tce_maxHitMatchedCounts_branch) {
      tce_maxHitMatchedCounts_branch->SetAddress(&tce_maxHitMatchedCounts_);
    }
  }
  t3_pt_branch = 0;
  if (tree->GetBranch("t3_pt") != 0) {
    t3_pt_branch = tree->GetBranch("t3_pt");
    if (t3_pt_branch) {
      t3_pt_branch->SetAddress(&t3_pt_);
    }
  }
  module_rings_branch = 0;
  if (tree->GetBranch("module_rings") != 0) {
    module_rings_branch = tree->GetBranch("module_rings");
    if (module_rings_branch) {
      module_rings_branch->SetAddress(&module_rings_);
    }
  }
  sim_T3_types_branch = 0;
  if (tree->GetBranch("sim_T3_types") != 0) {
    sim_T3_types_branch = tree->GetBranch("sim_T3_types");
    if (sim_T3_types_branch) {
      sim_T3_types_branch->SetAddress(&sim_T3_types_);
    }
  }
  sim_pT5_types_branch = 0;
  if (tree->GetBranch("sim_pT5_types") != 0) {
    sim_pT5_types_branch = tree->GetBranch("sim_pT5_types");
    if (sim_pT5_types_branch) {
      sim_pT5_types_branch->SetAddress(&sim_pT5_types_);
    }
  }
  sim_pT5_matched_branch = 0;
  if (tree->GetBranch("sim_pT5_matched") != 0) {
    sim_pT5_matched_branch = tree->GetBranch("sim_pT5_matched");
    if (sim_pT5_matched_branch) {
      sim_pT5_matched_branch->SetAddress(&sim_pT5_matched_);
    }
  }
  module_layers_branch = 0;
  if (tree->GetBranch("module_layers") != 0) {
    module_layers_branch = tree->GetBranch("module_layers");
    if (module_layers_branch) {
      module_layers_branch->SetAddress(&module_layers_);
    }
  }
  pT4_eta_branch = 0;
  if (tree->GetBranch("pT4_eta") != 0) {
    pT4_eta_branch = tree->GetBranch("pT4_eta");
    if (pT4_eta_branch) {
      pT4_eta_branch->SetAddress(&pT4_eta_);
    }
  }
  sim_tce_types_branch = 0;
  if (tree->GetBranch("sim_tce_types") != 0) {
    sim_tce_types_branch = tree->GetBranch("sim_tce_types");
    if (sim_tce_types_branch) {
      sim_tce_types_branch->SetAddress(&sim_tce_types_);
    }
  }
  tce_rzChiSquared_branch = 0;
  if (tree->GetBranch("tce_rzChiSquared") != 0) {
    tce_rzChiSquared_branch = tree->GetBranch("tce_rzChiSquared");
    if (tce_rzChiSquared_branch) {
      tce_rzChiSquared_branch->SetAddress(&tce_rzChiSquared_);
    }
  }
  pT3_matched_simIdx_branch = 0;
  if (tree->GetBranch("pT3_matched_simIdx") != 0) {
    pT3_matched_simIdx_branch = tree->GetBranch("pT3_matched_simIdx");
    if (pT3_matched_simIdx_branch) {
      pT3_matched_simIdx_branch->SetAddress(&pT3_matched_simIdx_);
    }
  }
  tree->SetMakeClass(0);
}
void LSTEff::GetEntry(unsigned int idx) {
  index = idx;
  pT5_occupancies_isLoaded = false;
  t3_phi_isLoaded = false;
  t5_score_rphisum_isLoaded = false;
  pT4_isFake_isLoaded = false;
  t3_isDuplicate_isLoaded = false;
  sim_event_isLoaded = false;
  sim_q_isLoaded = false;
  sim_eta_isLoaded = false;
  pT3_foundDuplicate_isLoaded = false;
  sim_len_isLoaded = false;
  pureTCE_isDuplicate_isLoaded = false;
  pT3_score_isLoaded = false;
  t5_eta_isLoaded = false;
  sim_denom_isLoaded = false;
  pT5_isDuplicate_isLoaded = false;
  sim_tce_matched_isLoaded = false;
  pT3_isDuplicate_isLoaded = false;
  tc_isDuplicate_isLoaded = false;
  pT3_eta_2_isLoaded = false;
  sim_pT3_matched_isLoaded = false;
  pureTCE_rzChiSquared_isLoaded = false;
  t4_isDuplicate_isLoaded = false;
  pureTCE_eta_isLoaded = false;
  tce_rPhiChiSquared_isLoaded = false;
  pureTCE_anchorType_isLoaded = false;
  pureTCE_pt_isLoaded = false;
  sim_pt_isLoaded = false;
  t5_eta_2_isLoaded = false;
  pLS_eta_isLoaded = false;
  sim_pdgId_isLoaded = false;
  t3_eta_isLoaded = false;
  tce_layer_binary_isLoaded = false;
  sim_TC_matched_nonextended_isLoaded = false;
  t4_occupancies_isLoaded = false;
  tce_eta_isLoaded = false;
  tce_isDuplicate_isLoaded = false;
  pT5_matched_simIdx_isLoaded = false;
  sim_tcIdx_isLoaded = false;
  t5_phi_2_isLoaded = false;
  pureTCE_maxHitMatchedCounts_isLoaded = false;
  t5_matched_simIdx_isLoaded = false;
  module_subdets_isLoaded = false;
  tce_anchorType_isLoaded = false;
  tce_nHitOverlaps_isLoaded = false;
  t3_isFake_isLoaded = false;
  tce_phi_isLoaded = false;
  t5_isFake_isLoaded = false;
  md_occupancies_isLoaded = false;
  t5_hitIdxs_isLoaded = false;
  sim_pT3_types_isLoaded = false;
  sim_pureTCE_types_isLoaded = false;
  t4_phi_isLoaded = false;
  t5_phi_isLoaded = false;
  pT5_hitIdxs_isLoaded = false;
  t5_pt_isLoaded = false;
  pT5_phi_isLoaded = false;
  pureTCE_isFake_isLoaded = false;
  tce_pt_isLoaded = false;
  tc_isFake_isLoaded = false;
  pT3_isFake_isLoaded = false;
  tce_nLayerOverlaps_isLoaded = false;
  tc_sim_isLoaded = false;
  sim_pLS_types_isLoaded = false;
  sim_pca_dxy_isLoaded = false;
  pT4_phi_isLoaded = false;
  sim_hits_isLoaded = false;
  pLS_phi_isLoaded = false;
  sim_pureTCE_matched_isLoaded = false;
  t3_occupancies_isLoaded = false;
  t5_foundDuplicate_isLoaded = false;
  sim_pT4_types_isLoaded = false;
  t4_isFake_isLoaded = false;
  simvtx_x_isLoaded = false;
  simvtx_y_isLoaded = false;
  simvtx_z_isLoaded = false;
  sim_T4_matched_isLoaded = false;
  sim_isGood_isLoaded = false;
  pT3_pt_isLoaded = false;
  tc_pt_isLoaded = false;
  pT3_phi_2_isLoaded = false;
  pT5_pt_isLoaded = false;
  pureTCE_rPhiChiSquared_isLoaded = false;
  pT5_score_isLoaded = false;
  sim_phi_isLoaded = false;
  pT5_isFake_isLoaded = false;
  tc_maxHitMatchedCounts_isLoaded = false;
  pureTCE_nLayerOverlaps_isLoaded = false;
  sim_pca_dz_isLoaded = false;
  pureTCE_hitIdxs_isLoaded = false;
  pureTCE_nHitOverlaps_isLoaded = false;
  sim_pLS_matched_isLoaded = false;
  tc_matched_simIdx_isLoaded = false;
  sim_T3_matched_isLoaded = false;
  pLS_score_isLoaded = false;
  pT3_phi_isLoaded = false;
  pT5_eta_isLoaded = false;
  tc_phi_isLoaded = false;
  t4_eta_isLoaded = false;
  pLS_isFake_isLoaded = false;
  pureTCE_matched_simIdx_isLoaded = false;
  sim_bunchCrossing_isLoaded = false;
  tc_partOfExtension_isLoaded = false;
  pT3_eta_isLoaded = false;
  sim_parentVtxIdx_isLoaded = false;
  pureTCE_layer_binary_isLoaded = false;
  sim_pT4_matched_isLoaded = false;
  tc_eta_isLoaded = false;
  sim_lengap_isLoaded = false;
  sim_T5_matched_isLoaded = false;
  sim_T5_types_isLoaded = false;
  tce_matched_simIdx_isLoaded = false;
  t5_isDuplicate_isLoaded = false;
  pT3_hitIdxs_isLoaded = false;
  tc_hitIdxs_isLoaded = false;
  pT3_occupancies_isLoaded = false;
  tc_occupancies_isLoaded = false;
  sim_TC_matched_isLoaded = false;
  sim_TC_matched_mask_isLoaded = false;
  pLS_isDuplicate_isLoaded = false;
  tce_anchorIndex_isLoaded = false;
  t5_occupancies_isLoaded = false;
  tc_type_isLoaded = false;
  tce_isFake_isLoaded = false;
  pLS_pt_isLoaded = false;
  pureTCE_anchorIndex_isLoaded = false;
  sim_T4_types_isLoaded = false;
  pT4_isDuplicate_isLoaded = false;
  t4_pt_isLoaded = false;
  sim_TC_types_isLoaded = false;
  sg_occupancies_isLoaded = false;
  pT4_pt_isLoaded = false;
  pureTCE_phi_isLoaded = false;
  sim_vx_isLoaded = false;
  sim_vy_isLoaded = false;
  sim_vz_isLoaded = false;
  tce_maxHitMatchedCounts_isLoaded = false;
  t3_pt_isLoaded = false;
  module_rings_isLoaded = false;
  sim_T3_types_isLoaded = false;
  sim_pT5_types_isLoaded = false;
  sim_pT5_matched_isLoaded = false;
  module_layers_isLoaded = false;
  pT4_eta_isLoaded = false;
  sim_tce_types_isLoaded = false;
  tce_rzChiSquared_isLoaded = false;
  pT3_matched_simIdx_isLoaded = false;
}
void LSTEff::LoadAllBranches() {
  if (pT5_occupancies_branch != 0)
    pT5_occupancies();
  if (t3_phi_branch != 0)
    t3_phi();
  if (t5_score_rphisum_branch != 0)
    t5_score_rphisum();
  if (pT4_isFake_branch != 0)
    pT4_isFake();
  if (t3_isDuplicate_branch != 0)
    t3_isDuplicate();
  if (sim_event_branch != 0)
    sim_event();
  if (sim_q_branch != 0)
    sim_q();
  if (sim_eta_branch != 0)
    sim_eta();
  if (pT3_foundDuplicate_branch != 0)
    pT3_foundDuplicate();
  if (sim_len_branch != 0)
    sim_len();
  if (pureTCE_isDuplicate_branch != 0)
    pureTCE_isDuplicate();
  if (pT3_score_branch != 0)
    pT3_score();
  if (t5_eta_branch != 0)
    t5_eta();
  if (sim_denom_branch != 0)
    sim_denom();
  if (pT5_isDuplicate_branch != 0)
    pT5_isDuplicate();
  if (sim_tce_matched_branch != 0)
    sim_tce_matched();
  if (pT3_isDuplicate_branch != 0)
    pT3_isDuplicate();
  if (tc_isDuplicate_branch != 0)
    tc_isDuplicate();
  if (pT3_eta_2_branch != 0)
    pT3_eta_2();
  if (sim_pT3_matched_branch != 0)
    sim_pT3_matched();
  if (pureTCE_rzChiSquared_branch != 0)
    pureTCE_rzChiSquared();
  if (t4_isDuplicate_branch != 0)
    t4_isDuplicate();
  if (pureTCE_eta_branch != 0)
    pureTCE_eta();
  if (tce_rPhiChiSquared_branch != 0)
    tce_rPhiChiSquared();
  if (pureTCE_anchorType_branch != 0)
    pureTCE_anchorType();
  if (pureTCE_pt_branch != 0)
    pureTCE_pt();
  if (sim_pt_branch != 0)
    sim_pt();
  if (t5_eta_2_branch != 0)
    t5_eta_2();
  if (pLS_eta_branch != 0)
    pLS_eta();
  if (sim_pdgId_branch != 0)
    sim_pdgId();
  if (t3_eta_branch != 0)
    t3_eta();
  if (tce_layer_binary_branch != 0)
    tce_layer_binary();
  if (sim_TC_matched_nonextended_branch != 0)
    sim_TC_matched_nonextended();
  if (t4_occupancies_branch != 0)
    t4_occupancies();
  if (tce_eta_branch != 0)
    tce_eta();
  if (tce_isDuplicate_branch != 0)
    tce_isDuplicate();
  if (pT5_matched_simIdx_branch != 0)
    pT5_matched_simIdx();
  if (sim_tcIdx_branch != 0)
    sim_tcIdx();
  if (t5_phi_2_branch != 0)
    t5_phi_2();
  if (pureTCE_maxHitMatchedCounts_branch != 0)
    pureTCE_maxHitMatchedCounts();
  if (t5_matched_simIdx_branch != 0)
    t5_matched_simIdx();
  if (module_subdets_branch != 0)
    module_subdets();
  if (tce_anchorType_branch != 0)
    tce_anchorType();
  if (tce_nHitOverlaps_branch != 0)
    tce_nHitOverlaps();
  if (t3_isFake_branch != 0)
    t3_isFake();
  if (tce_phi_branch != 0)
    tce_phi();
  if (t5_isFake_branch != 0)
    t5_isFake();
  if (md_occupancies_branch != 0)
    md_occupancies();
  if (t5_hitIdxs_branch != 0)
    t5_hitIdxs();
  if (sim_pT3_types_branch != 0)
    sim_pT3_types();
  if (sim_pureTCE_types_branch != 0)
    sim_pureTCE_types();
  if (t4_phi_branch != 0)
    t4_phi();
  if (t5_phi_branch != 0)
    t5_phi();
  if (pT5_hitIdxs_branch != 0)
    pT5_hitIdxs();
  if (t5_pt_branch != 0)
    t5_pt();
  if (pT5_phi_branch != 0)
    pT5_phi();
  if (pureTCE_isFake_branch != 0)
    pureTCE_isFake();
  if (tce_pt_branch != 0)
    tce_pt();
  if (tc_isFake_branch != 0)
    tc_isFake();
  if (pT3_isFake_branch != 0)
    pT3_isFake();
  if (tce_nLayerOverlaps_branch != 0)
    tce_nLayerOverlaps();
  if (tc_sim_branch != 0)
    tc_sim();
  if (sim_pLS_types_branch != 0)
    sim_pLS_types();
  if (sim_pca_dxy_branch != 0)
    sim_pca_dxy();
  if (pT4_phi_branch != 0)
    pT4_phi();
  if (sim_hits_branch != 0)
    sim_hits();
  if (pLS_phi_branch != 0)
    pLS_phi();
  if (sim_pureTCE_matched_branch != 0)
    sim_pureTCE_matched();
  if (t3_occupancies_branch != 0)
    t3_occupancies();
  if (t5_foundDuplicate_branch != 0)
    t5_foundDuplicate();
  if (sim_pT4_types_branch != 0)
    sim_pT4_types();
  if (t4_isFake_branch != 0)
    t4_isFake();
  if (simvtx_x_branch != 0)
    simvtx_x();
  if (simvtx_y_branch != 0)
    simvtx_y();
  if (simvtx_z_branch != 0)
    simvtx_z();
  if (sim_T4_matched_branch != 0)
    sim_T4_matched();
  if (sim_isGood_branch != 0)
    sim_isGood();
  if (pT3_pt_branch != 0)
    pT3_pt();
  if (tc_pt_branch != 0)
    tc_pt();
  if (pT3_phi_2_branch != 0)
    pT3_phi_2();
  if (pT5_pt_branch != 0)
    pT5_pt();
  if (pureTCE_rPhiChiSquared_branch != 0)
    pureTCE_rPhiChiSquared();
  if (pT5_score_branch != 0)
    pT5_score();
  if (sim_phi_branch != 0)
    sim_phi();
  if (pT5_isFake_branch != 0)
    pT5_isFake();
  if (tc_maxHitMatchedCounts_branch != 0)
    tc_maxHitMatchedCounts();
  if (pureTCE_nLayerOverlaps_branch != 0)
    pureTCE_nLayerOverlaps();
  if (sim_pca_dz_branch != 0)
    sim_pca_dz();
  if (pureTCE_hitIdxs_branch != 0)
    pureTCE_hitIdxs();
  if (pureTCE_nHitOverlaps_branch != 0)
    pureTCE_nHitOverlaps();
  if (sim_pLS_matched_branch != 0)
    sim_pLS_matched();
  if (tc_matched_simIdx_branch != 0)
    tc_matched_simIdx();
  if (sim_T3_matched_branch != 0)
    sim_T3_matched();
  if (pLS_score_branch != 0)
    pLS_score();
  if (pT3_phi_branch != 0)
    pT3_phi();
  if (pT5_eta_branch != 0)
    pT5_eta();
  if (tc_phi_branch != 0)
    tc_phi();
  if (t4_eta_branch != 0)
    t4_eta();
  if (pLS_isFake_branch != 0)
    pLS_isFake();
  if (pureTCE_matched_simIdx_branch != 0)
    pureTCE_matched_simIdx();
  if (sim_bunchCrossing_branch != 0)
    sim_bunchCrossing();
  if (tc_partOfExtension_branch != 0)
    tc_partOfExtension();
  if (pT3_eta_branch != 0)
    pT3_eta();
  if (sim_parentVtxIdx_branch != 0)
    sim_parentVtxIdx();
  if (pureTCE_layer_binary_branch != 0)
    pureTCE_layer_binary();
  if (sim_pT4_matched_branch != 0)
    sim_pT4_matched();
  if (tc_eta_branch != 0)
    tc_eta();
  if (sim_lengap_branch != 0)
    sim_lengap();
  if (sim_T5_matched_branch != 0)
    sim_T5_matched();
  if (sim_T5_types_branch != 0)
    sim_T5_types();
  if (tce_matched_simIdx_branch != 0)
    tce_matched_simIdx();
  if (t5_isDuplicate_branch != 0)
    t5_isDuplicate();
  if (pT3_hitIdxs_branch != 0)
    pT3_hitIdxs();
  if (tc_hitIdxs_branch != 0)
    tc_hitIdxs();
  if (pT3_occupancies_branch != 0)
    pT3_occupancies();
  if (tc_occupancies_branch != 0)
    tc_occupancies();
  if (sim_TC_matched_branch != 0)
    sim_TC_matched();
  if (sim_TC_matched_mask_branch != 0)
    sim_TC_matched_mask();
  if (pLS_isDuplicate_branch != 0)
    pLS_isDuplicate();
  if (tce_anchorIndex_branch != 0)
    tce_anchorIndex();
  if (t5_occupancies_branch != 0)
    t5_occupancies();
  if (tc_type_branch != 0)
    tc_type();
  if (tce_isFake_branch != 0)
    tce_isFake();
  if (pLS_pt_branch != 0)
    pLS_pt();
  if (pureTCE_anchorIndex_branch != 0)
    pureTCE_anchorIndex();
  if (sim_T4_types_branch != 0)
    sim_T4_types();
  if (pT4_isDuplicate_branch != 0)
    pT4_isDuplicate();
  if (t4_pt_branch != 0)
    t4_pt();
  if (sim_TC_types_branch != 0)
    sim_TC_types();
  if (sg_occupancies_branch != 0)
    sg_occupancies();
  if (pT4_pt_branch != 0)
    pT4_pt();
  if (pureTCE_phi_branch != 0)
    pureTCE_phi();
  if (sim_vx_branch != 0)
    sim_vx();
  if (sim_vy_branch != 0)
    sim_vy();
  if (sim_vz_branch != 0)
    sim_vz();
  if (tce_maxHitMatchedCounts_branch != 0)
    tce_maxHitMatchedCounts();
  if (t3_pt_branch != 0)
    t3_pt();
  if (module_rings_branch != 0)
    module_rings();
  if (sim_T3_types_branch != 0)
    sim_T3_types();
  if (sim_pT5_types_branch != 0)
    sim_pT5_types();
  if (sim_pT5_matched_branch != 0)
    sim_pT5_matched();
  if (module_layers_branch != 0)
    module_layers();
  if (pT4_eta_branch != 0)
    pT4_eta();
  if (sim_tce_types_branch != 0)
    sim_tce_types();
  if (tce_rzChiSquared_branch != 0)
    tce_rzChiSquared();
  if (pT3_matched_simIdx_branch != 0)
    pT3_matched_simIdx();
}
const int &LSTEff::pT5_occupancies() {
  if (not pT5_occupancies_isLoaded) {
    if (pT5_occupancies_branch != 0) {
      pT5_occupancies_branch->GetEntry(index);
    } else {
      printf("branch pT5_occupancies_branch does not exist!\n");
      exit(1);
    }
    pT5_occupancies_isLoaded = true;
  }
  return pT5_occupancies_;
}
const std::vector<float> &LSTEff::t3_phi() {
  if (not t3_phi_isLoaded) {
    if (t3_phi_branch != 0) {
      t3_phi_branch->GetEntry(index);
    } else {
      printf("branch t3_phi_branch does not exist!\n");
      exit(1);
    }
    t3_phi_isLoaded = true;
  }
  return *t3_phi_;
}
const std::vector<float> &LSTEff::t5_score_rphisum() {
  if (not t5_score_rphisum_isLoaded) {
    if (t5_score_rphisum_branch != 0) {
      t5_score_rphisum_branch->GetEntry(index);
    } else {
      printf("branch t5_score_rphisum_branch does not exist!\n");
      exit(1);
    }
    t5_score_rphisum_isLoaded = true;
  }
  return *t5_score_rphisum_;
}
const std::vector<int> &LSTEff::pT4_isFake() {
  if (not pT4_isFake_isLoaded) {
    if (pT4_isFake_branch != 0) {
      pT4_isFake_branch->GetEntry(index);
    } else {
      printf("branch pT4_isFake_branch does not exist!\n");
      exit(1);
    }
    pT4_isFake_isLoaded = true;
  }
  return *pT4_isFake_;
}
const std::vector<int> &LSTEff::t3_isDuplicate() {
  if (not t3_isDuplicate_isLoaded) {
    if (t3_isDuplicate_branch != 0) {
      t3_isDuplicate_branch->GetEntry(index);
    } else {
      printf("branch t3_isDuplicate_branch does not exist!\n");
      exit(1);
    }
    t3_isDuplicate_isLoaded = true;
  }
  return *t3_isDuplicate_;
}
const std::vector<int> &LSTEff::sim_event() {
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
const std::vector<int> &LSTEff::sim_q() {
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
const std::vector<float> &LSTEff::sim_eta() {
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
const std::vector<int> &LSTEff::pT3_foundDuplicate() {
  if (not pT3_foundDuplicate_isLoaded) {
    if (pT3_foundDuplicate_branch != 0) {
      pT3_foundDuplicate_branch->GetEntry(index);
    } else {
      printf("branch pT3_foundDuplicate_branch does not exist!\n");
      exit(1);
    }
    pT3_foundDuplicate_isLoaded = true;
  }
  return *pT3_foundDuplicate_;
}
const std::vector<float> &LSTEff::sim_len() {
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
const std::vector<int> &LSTEff::pureTCE_isDuplicate() {
  if (not pureTCE_isDuplicate_isLoaded) {
    if (pureTCE_isDuplicate_branch != 0) {
      pureTCE_isDuplicate_branch->GetEntry(index);
    } else {
      printf("branch pureTCE_isDuplicate_branch does not exist!\n");
      exit(1);
    }
    pureTCE_isDuplicate_isLoaded = true;
  }
  return *pureTCE_isDuplicate_;
}
const std::vector<float> &LSTEff::pT3_score() {
  if (not pT3_score_isLoaded) {
    if (pT3_score_branch != 0) {
      pT3_score_branch->GetEntry(index);
    } else {
      printf("branch pT3_score_branch does not exist!\n");
      exit(1);
    }
    pT3_score_isLoaded = true;
  }
  return *pT3_score_;
}
const std::vector<float> &LSTEff::t5_eta() {
  if (not t5_eta_isLoaded) {
    if (t5_eta_branch != 0) {
      t5_eta_branch->GetEntry(index);
    } else {
      printf("branch t5_eta_branch does not exist!\n");
      exit(1);
    }
    t5_eta_isLoaded = true;
  }
  return *t5_eta_;
}
const std::vector<int> &LSTEff::sim_denom() {
  if (not sim_denom_isLoaded) {
    if (sim_denom_branch != 0) {
      sim_denom_branch->GetEntry(index);
    } else {
      printf("branch sim_denom_branch does not exist!\n");
      exit(1);
    }
    sim_denom_isLoaded = true;
  }
  return *sim_denom_;
}
const std::vector<int> &LSTEff::pT5_isDuplicate() {
  if (not pT5_isDuplicate_isLoaded) {
    if (pT5_isDuplicate_branch != 0) {
      pT5_isDuplicate_branch->GetEntry(index);
    } else {
      printf("branch pT5_isDuplicate_branch does not exist!\n");
      exit(1);
    }
    pT5_isDuplicate_isLoaded = true;
  }
  return *pT5_isDuplicate_;
}
const std::vector<int> &LSTEff::sim_tce_matched() {
  if (not sim_tce_matched_isLoaded) {
    if (sim_tce_matched_branch != 0) {
      sim_tce_matched_branch->GetEntry(index);
    } else {
      printf("branch sim_tce_matched_branch does not exist!\n");
      exit(1);
    }
    sim_tce_matched_isLoaded = true;
  }
  return *sim_tce_matched_;
}
const std::vector<int> &LSTEff::pT3_isDuplicate() {
  if (not pT3_isDuplicate_isLoaded) {
    if (pT3_isDuplicate_branch != 0) {
      pT3_isDuplicate_branch->GetEntry(index);
    } else {
      printf("branch pT3_isDuplicate_branch does not exist!\n");
      exit(1);
    }
    pT3_isDuplicate_isLoaded = true;
  }
  return *pT3_isDuplicate_;
}
const std::vector<int> &LSTEff::tc_isDuplicate() {
  if (not tc_isDuplicate_isLoaded) {
    if (tc_isDuplicate_branch != 0) {
      tc_isDuplicate_branch->GetEntry(index);
    } else {
      printf("branch tc_isDuplicate_branch does not exist!\n");
      exit(1);
    }
    tc_isDuplicate_isLoaded = true;
  }
  return *tc_isDuplicate_;
}
const std::vector<float> &LSTEff::pT3_eta_2() {
  if (not pT3_eta_2_isLoaded) {
    if (pT3_eta_2_branch != 0) {
      pT3_eta_2_branch->GetEntry(index);
    } else {
      printf("branch pT3_eta_2_branch does not exist!\n");
      exit(1);
    }
    pT3_eta_2_isLoaded = true;
  }
  return *pT3_eta_2_;
}
const std::vector<int> &LSTEff::sim_pT3_matched() {
  if (not sim_pT3_matched_isLoaded) {
    if (sim_pT3_matched_branch != 0) {
      sim_pT3_matched_branch->GetEntry(index);
    } else {
      printf("branch sim_pT3_matched_branch does not exist!\n");
      exit(1);
    }
    sim_pT3_matched_isLoaded = true;
  }
  return *sim_pT3_matched_;
}
const std::vector<float> &LSTEff::pureTCE_rzChiSquared() {
  if (not pureTCE_rzChiSquared_isLoaded) {
    if (pureTCE_rzChiSquared_branch != 0) {
      pureTCE_rzChiSquared_branch->GetEntry(index);
    } else {
      printf("branch pureTCE_rzChiSquared_branch does not exist!\n");
      exit(1);
    }
    pureTCE_rzChiSquared_isLoaded = true;
  }
  return *pureTCE_rzChiSquared_;
}
const std::vector<int> &LSTEff::t4_isDuplicate() {
  if (not t4_isDuplicate_isLoaded) {
    if (t4_isDuplicate_branch != 0) {
      t4_isDuplicate_branch->GetEntry(index);
    } else {
      printf("branch t4_isDuplicate_branch does not exist!\n");
      exit(1);
    }
    t4_isDuplicate_isLoaded = true;
  }
  return *t4_isDuplicate_;
}
const std::vector<float> &LSTEff::pureTCE_eta() {
  if (not pureTCE_eta_isLoaded) {
    if (pureTCE_eta_branch != 0) {
      pureTCE_eta_branch->GetEntry(index);
    } else {
      printf("branch pureTCE_eta_branch does not exist!\n");
      exit(1);
    }
    pureTCE_eta_isLoaded = true;
  }
  return *pureTCE_eta_;
}
const std::vector<float> &LSTEff::tce_rPhiChiSquared() {
  if (not tce_rPhiChiSquared_isLoaded) {
    if (tce_rPhiChiSquared_branch != 0) {
      tce_rPhiChiSquared_branch->GetEntry(index);
    } else {
      printf("branch tce_rPhiChiSquared_branch does not exist!\n");
      exit(1);
    }
    tce_rPhiChiSquared_isLoaded = true;
  }
  return *tce_rPhiChiSquared_;
}
const std::vector<int> &LSTEff::pureTCE_anchorType() {
  if (not pureTCE_anchorType_isLoaded) {
    if (pureTCE_anchorType_branch != 0) {
      pureTCE_anchorType_branch->GetEntry(index);
    } else {
      printf("branch pureTCE_anchorType_branch does not exist!\n");
      exit(1);
    }
    pureTCE_anchorType_isLoaded = true;
  }
  return *pureTCE_anchorType_;
}
const std::vector<float> &LSTEff::pureTCE_pt() {
  if (not pureTCE_pt_isLoaded) {
    if (pureTCE_pt_branch != 0) {
      pureTCE_pt_branch->GetEntry(index);
    } else {
      printf("branch pureTCE_pt_branch does not exist!\n");
      exit(1);
    }
    pureTCE_pt_isLoaded = true;
  }
  return *pureTCE_pt_;
}
const std::vector<float> &LSTEff::sim_pt() {
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
const std::vector<float> &LSTEff::t5_eta_2() {
  if (not t5_eta_2_isLoaded) {
    if (t5_eta_2_branch != 0) {
      t5_eta_2_branch->GetEntry(index);
    } else {
      printf("branch t5_eta_2_branch does not exist!\n");
      exit(1);
    }
    t5_eta_2_isLoaded = true;
  }
  return *t5_eta_2_;
}
const std::vector<float> &LSTEff::pLS_eta() {
  if (not pLS_eta_isLoaded) {
    if (pLS_eta_branch != 0) {
      pLS_eta_branch->GetEntry(index);
    } else {
      printf("branch pLS_eta_branch does not exist!\n");
      exit(1);
    }
    pLS_eta_isLoaded = true;
  }
  return *pLS_eta_;
}
const std::vector<int> &LSTEff::sim_pdgId() {
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
const std::vector<float> &LSTEff::t3_eta() {
  if (not t3_eta_isLoaded) {
    if (t3_eta_branch != 0) {
      t3_eta_branch->GetEntry(index);
    } else {
      printf("branch t3_eta_branch does not exist!\n");
      exit(1);
    }
    t3_eta_isLoaded = true;
  }
  return *t3_eta_;
}
const std::vector<int> &LSTEff::tce_layer_binary() {
  if (not tce_layer_binary_isLoaded) {
    if (tce_layer_binary_branch != 0) {
      tce_layer_binary_branch->GetEntry(index);
    } else {
      printf("branch tce_layer_binary_branch does not exist!\n");
      exit(1);
    }
    tce_layer_binary_isLoaded = true;
  }
  return *tce_layer_binary_;
}
const std::vector<int> &LSTEff::sim_TC_matched_nonextended() {
  if (not sim_TC_matched_nonextended_isLoaded) {
    if (sim_TC_matched_nonextended_branch != 0) {
      sim_TC_matched_nonextended_branch->GetEntry(index);
    } else {
      printf("branch sim_TC_matched_nonextended_branch does not exist!\n");
      exit(1);
    }
    sim_TC_matched_nonextended_isLoaded = true;
  }
  return *sim_TC_matched_nonextended_;
}
const std::vector<int> &LSTEff::t4_occupancies() {
  if (not t4_occupancies_isLoaded) {
    if (t4_occupancies_branch != 0) {
      t4_occupancies_branch->GetEntry(index);
    } else {
      printf("branch t4_occupancies_branch does not exist!\n");
      exit(1);
    }
    t4_occupancies_isLoaded = true;
  }
  return *t4_occupancies_;
}
const std::vector<float> &LSTEff::tce_eta() {
  if (not tce_eta_isLoaded) {
    if (tce_eta_branch != 0) {
      tce_eta_branch->GetEntry(index);
    } else {
      printf("branch tce_eta_branch does not exist!\n");
      exit(1);
    }
    tce_eta_isLoaded = true;
  }
  return *tce_eta_;
}
const std::vector<int> &LSTEff::tce_isDuplicate() {
  if (not tce_isDuplicate_isLoaded) {
    if (tce_isDuplicate_branch != 0) {
      tce_isDuplicate_branch->GetEntry(index);
    } else {
      printf("branch tce_isDuplicate_branch does not exist!\n");
      exit(1);
    }
    tce_isDuplicate_isLoaded = true;
  }
  return *tce_isDuplicate_;
}
const std::vector<std::vector<int> > &LSTEff::pT5_matched_simIdx() {
  if (not pT5_matched_simIdx_isLoaded) {
    if (pT5_matched_simIdx_branch != 0) {
      pT5_matched_simIdx_branch->GetEntry(index);
    } else {
      printf("branch pT5_matched_simIdx_branch does not exist!\n");
      exit(1);
    }
    pT5_matched_simIdx_isLoaded = true;
  }
  return *pT5_matched_simIdx_;
}
const std::vector<std::vector<int> > &LSTEff::sim_tcIdx() {
  if (not sim_tcIdx_isLoaded) {
    if (sim_tcIdx_branch != 0) {
      sim_tcIdx_branch->GetEntry(index);
    } else {
      printf("branch sim_tcIdx_branch does not exist!\n");
      exit(1);
    }
    sim_tcIdx_isLoaded = true;
  }
  return *sim_tcIdx_;
}
const std::vector<float> &LSTEff::t5_phi_2() {
  if (not t5_phi_2_isLoaded) {
    if (t5_phi_2_branch != 0) {
      t5_phi_2_branch->GetEntry(index);
    } else {
      printf("branch t5_phi_2_branch does not exist!\n");
      exit(1);
    }
    t5_phi_2_isLoaded = true;
  }
  return *t5_phi_2_;
}
const std::vector<int> &LSTEff::pureTCE_maxHitMatchedCounts() {
  if (not pureTCE_maxHitMatchedCounts_isLoaded) {
    if (pureTCE_maxHitMatchedCounts_branch != 0) {
      pureTCE_maxHitMatchedCounts_branch->GetEntry(index);
    } else {
      printf("branch pureTCE_maxHitMatchedCounts_branch does not exist!\n");
      exit(1);
    }
    pureTCE_maxHitMatchedCounts_isLoaded = true;
  }
  return *pureTCE_maxHitMatchedCounts_;
}
const std::vector<std::vector<int> > &LSTEff::t5_matched_simIdx() {
  if (not t5_matched_simIdx_isLoaded) {
    if (t5_matched_simIdx_branch != 0) {
      t5_matched_simIdx_branch->GetEntry(index);
    } else {
      printf("branch t5_matched_simIdx_branch does not exist!\n");
      exit(1);
    }
    t5_matched_simIdx_isLoaded = true;
  }
  return *t5_matched_simIdx_;
}
const std::vector<int> &LSTEff::module_subdets() {
  if (not module_subdets_isLoaded) {
    if (module_subdets_branch != 0) {
      module_subdets_branch->GetEntry(index);
    } else {
      printf("branch module_subdets_branch does not exist!\n");
      exit(1);
    }
    module_subdets_isLoaded = true;
  }
  return *module_subdets_;
}
const std::vector<int> &LSTEff::tce_anchorType() {
  if (not tce_anchorType_isLoaded) {
    if (tce_anchorType_branch != 0) {
      tce_anchorType_branch->GetEntry(index);
    } else {
      printf("branch tce_anchorType_branch does not exist!\n");
      exit(1);
    }
    tce_anchorType_isLoaded = true;
  }
  return *tce_anchorType_;
}
const std::vector<std::vector<int> > &LSTEff::tce_nHitOverlaps() {
  if (not tce_nHitOverlaps_isLoaded) {
    if (tce_nHitOverlaps_branch != 0) {
      tce_nHitOverlaps_branch->GetEntry(index);
    } else {
      printf("branch tce_nHitOverlaps_branch does not exist!\n");
      exit(1);
    }
    tce_nHitOverlaps_isLoaded = true;
  }
  return *tce_nHitOverlaps_;
}
const std::vector<int> &LSTEff::t3_isFake() {
  if (not t3_isFake_isLoaded) {
    if (t3_isFake_branch != 0) {
      t3_isFake_branch->GetEntry(index);
    } else {
      printf("branch t3_isFake_branch does not exist!\n");
      exit(1);
    }
    t3_isFake_isLoaded = true;
  }
  return *t3_isFake_;
}
const std::vector<float> &LSTEff::tce_phi() {
  if (not tce_phi_isLoaded) {
    if (tce_phi_branch != 0) {
      tce_phi_branch->GetEntry(index);
    } else {
      printf("branch tce_phi_branch does not exist!\n");
      exit(1);
    }
    tce_phi_isLoaded = true;
  }
  return *tce_phi_;
}
const std::vector<int> &LSTEff::t5_isFake() {
  if (not t5_isFake_isLoaded) {
    if (t5_isFake_branch != 0) {
      t5_isFake_branch->GetEntry(index);
    } else {
      printf("branch t5_isFake_branch does not exist!\n");
      exit(1);
    }
    t5_isFake_isLoaded = true;
  }
  return *t5_isFake_;
}
const std::vector<int> &LSTEff::md_occupancies() {
  if (not md_occupancies_isLoaded) {
    if (md_occupancies_branch != 0) {
      md_occupancies_branch->GetEntry(index);
    } else {
      printf("branch md_occupancies_branch does not exist!\n");
      exit(1);
    }
    md_occupancies_isLoaded = true;
  }
  return *md_occupancies_;
}
const std::vector<std::vector<int> > &LSTEff::t5_hitIdxs() {
  if (not t5_hitIdxs_isLoaded) {
    if (t5_hitIdxs_branch != 0) {
      t5_hitIdxs_branch->GetEntry(index);
    } else {
      printf("branch t5_hitIdxs_branch does not exist!\n");
      exit(1);
    }
    t5_hitIdxs_isLoaded = true;
  }
  return *t5_hitIdxs_;
}
const std::vector<std::vector<int> > &LSTEff::sim_pT3_types() {
  if (not sim_pT3_types_isLoaded) {
    if (sim_pT3_types_branch != 0) {
      sim_pT3_types_branch->GetEntry(index);
    } else {
      printf("branch sim_pT3_types_branch does not exist!\n");
      exit(1);
    }
    sim_pT3_types_isLoaded = true;
  }
  return *sim_pT3_types_;
}
const std::vector<std::vector<int> > &LSTEff::sim_pureTCE_types() {
  if (not sim_pureTCE_types_isLoaded) {
    if (sim_pureTCE_types_branch != 0) {
      sim_pureTCE_types_branch->GetEntry(index);
    } else {
      printf("branch sim_pureTCE_types_branch does not exist!\n");
      exit(1);
    }
    sim_pureTCE_types_isLoaded = true;
  }
  return *sim_pureTCE_types_;
}
const std::vector<float> &LSTEff::t4_phi() {
  if (not t4_phi_isLoaded) {
    if (t4_phi_branch != 0) {
      t4_phi_branch->GetEntry(index);
    } else {
      printf("branch t4_phi_branch does not exist!\n");
      exit(1);
    }
    t4_phi_isLoaded = true;
  }
  return *t4_phi_;
}
const std::vector<float> &LSTEff::t5_phi() {
  if (not t5_phi_isLoaded) {
    if (t5_phi_branch != 0) {
      t5_phi_branch->GetEntry(index);
    } else {
      printf("branch t5_phi_branch does not exist!\n");
      exit(1);
    }
    t5_phi_isLoaded = true;
  }
  return *t5_phi_;
}
const std::vector<std::vector<int> > &LSTEff::pT5_hitIdxs() {
  if (not pT5_hitIdxs_isLoaded) {
    if (pT5_hitIdxs_branch != 0) {
      pT5_hitIdxs_branch->GetEntry(index);
    } else {
      printf("branch pT5_hitIdxs_branch does not exist!\n");
      exit(1);
    }
    pT5_hitIdxs_isLoaded = true;
  }
  return *pT5_hitIdxs_;
}
const std::vector<float> &LSTEff::t5_pt() {
  if (not t5_pt_isLoaded) {
    if (t5_pt_branch != 0) {
      t5_pt_branch->GetEntry(index);
    } else {
      printf("branch t5_pt_branch does not exist!\n");
      exit(1);
    }
    t5_pt_isLoaded = true;
  }
  return *t5_pt_;
}
const std::vector<float> &LSTEff::pT5_phi() {
  if (not pT5_phi_isLoaded) {
    if (pT5_phi_branch != 0) {
      pT5_phi_branch->GetEntry(index);
    } else {
      printf("branch pT5_phi_branch does not exist!\n");
      exit(1);
    }
    pT5_phi_isLoaded = true;
  }
  return *pT5_phi_;
}
const std::vector<int> &LSTEff::pureTCE_isFake() {
  if (not pureTCE_isFake_isLoaded) {
    if (pureTCE_isFake_branch != 0) {
      pureTCE_isFake_branch->GetEntry(index);
    } else {
      printf("branch pureTCE_isFake_branch does not exist!\n");
      exit(1);
    }
    pureTCE_isFake_isLoaded = true;
  }
  return *pureTCE_isFake_;
}
const std::vector<float> &LSTEff::tce_pt() {
  if (not tce_pt_isLoaded) {
    if (tce_pt_branch != 0) {
      tce_pt_branch->GetEntry(index);
    } else {
      printf("branch tce_pt_branch does not exist!\n");
      exit(1);
    }
    tce_pt_isLoaded = true;
  }
  return *tce_pt_;
}
const std::vector<int> &LSTEff::tc_isFake() {
  if (not tc_isFake_isLoaded) {
    if (tc_isFake_branch != 0) {
      tc_isFake_branch->GetEntry(index);
    } else {
      printf("branch tc_isFake_branch does not exist!\n");
      exit(1);
    }
    tc_isFake_isLoaded = true;
  }
  return *tc_isFake_;
}
const std::vector<int> &LSTEff::pT3_isFake() {
  if (not pT3_isFake_isLoaded) {
    if (pT3_isFake_branch != 0) {
      pT3_isFake_branch->GetEntry(index);
    } else {
      printf("branch pT3_isFake_branch does not exist!\n");
      exit(1);
    }
    pT3_isFake_isLoaded = true;
  }
  return *pT3_isFake_;
}
const std::vector<std::vector<int> > &LSTEff::tce_nLayerOverlaps() {
  if (not tce_nLayerOverlaps_isLoaded) {
    if (tce_nLayerOverlaps_branch != 0) {
      tce_nLayerOverlaps_branch->GetEntry(index);
    } else {
      printf("branch tce_nLayerOverlaps_branch does not exist!\n");
      exit(1);
    }
    tce_nLayerOverlaps_isLoaded = true;
  }
  return *tce_nLayerOverlaps_;
}
const std::vector<int> &LSTEff::tc_sim() {
  if (not tc_sim_isLoaded) {
    if (tc_sim_branch != 0) {
      tc_sim_branch->GetEntry(index);
    } else {
      printf("branch tc_sim_branch does not exist!\n");
      exit(1);
    }
    tc_sim_isLoaded = true;
  }
  return *tc_sim_;
}
const std::vector<std::vector<int> > &LSTEff::sim_pLS_types() {
  if (not sim_pLS_types_isLoaded) {
    if (sim_pLS_types_branch != 0) {
      sim_pLS_types_branch->GetEntry(index);
    } else {
      printf("branch sim_pLS_types_branch does not exist!\n");
      exit(1);
    }
    sim_pLS_types_isLoaded = true;
  }
  return *sim_pLS_types_;
}
const std::vector<float> &LSTEff::sim_pca_dxy() {
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
const std::vector<float> &LSTEff::pT4_phi() {
  if (not pT4_phi_isLoaded) {
    if (pT4_phi_branch != 0) {
      pT4_phi_branch->GetEntry(index);
    } else {
      printf("branch pT4_phi_branch does not exist!\n");
      exit(1);
    }
    pT4_phi_isLoaded = true;
  }
  return *pT4_phi_;
}
const std::vector<float> &LSTEff::sim_hits() {
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
const std::vector<float> &LSTEff::pLS_phi() {
  if (not pLS_phi_isLoaded) {
    if (pLS_phi_branch != 0) {
      pLS_phi_branch->GetEntry(index);
    } else {
      printf("branch pLS_phi_branch does not exist!\n");
      exit(1);
    }
    pLS_phi_isLoaded = true;
  }
  return *pLS_phi_;
}
const std::vector<int> &LSTEff::sim_pureTCE_matched() {
  if (not sim_pureTCE_matched_isLoaded) {
    if (sim_pureTCE_matched_branch != 0) {
      sim_pureTCE_matched_branch->GetEntry(index);
    } else {
      printf("branch sim_pureTCE_matched_branch does not exist!\n");
      exit(1);
    }
    sim_pureTCE_matched_isLoaded = true;
  }
  return *sim_pureTCE_matched_;
}
const std::vector<int> &LSTEff::t3_occupancies() {
  if (not t3_occupancies_isLoaded) {
    if (t3_occupancies_branch != 0) {
      t3_occupancies_branch->GetEntry(index);
    } else {
      printf("branch t3_occupancies_branch does not exist!\n");
      exit(1);
    }
    t3_occupancies_isLoaded = true;
  }
  return *t3_occupancies_;
}
const std::vector<int> &LSTEff::t5_foundDuplicate() {
  if (not t5_foundDuplicate_isLoaded) {
    if (t5_foundDuplicate_branch != 0) {
      t5_foundDuplicate_branch->GetEntry(index);
    } else {
      printf("branch t5_foundDuplicate_branch does not exist!\n");
      exit(1);
    }
    t5_foundDuplicate_isLoaded = true;
  }
  return *t5_foundDuplicate_;
}
const std::vector<std::vector<int> > &LSTEff::sim_pT4_types() {
  if (not sim_pT4_types_isLoaded) {
    if (sim_pT4_types_branch != 0) {
      sim_pT4_types_branch->GetEntry(index);
    } else {
      printf("branch sim_pT4_types_branch does not exist!\n");
      exit(1);
    }
    sim_pT4_types_isLoaded = true;
  }
  return *sim_pT4_types_;
}
const std::vector<int> &LSTEff::t4_isFake() {
  if (not t4_isFake_isLoaded) {
    if (t4_isFake_branch != 0) {
      t4_isFake_branch->GetEntry(index);
    } else {
      printf("branch t4_isFake_branch does not exist!\n");
      exit(1);
    }
    t4_isFake_isLoaded = true;
  }
  return *t4_isFake_;
}
const std::vector<float> &LSTEff::simvtx_x() {
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
const std::vector<float> &LSTEff::simvtx_y() {
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
const std::vector<float> &LSTEff::simvtx_z() {
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
const std::vector<int> &LSTEff::sim_T4_matched() {
  if (not sim_T4_matched_isLoaded) {
    if (sim_T4_matched_branch != 0) {
      sim_T4_matched_branch->GetEntry(index);
    } else {
      printf("branch sim_T4_matched_branch does not exist!\n");
      exit(1);
    }
    sim_T4_matched_isLoaded = true;
  }
  return *sim_T4_matched_;
}
const std::vector<bool> &LSTEff::sim_isGood() {
  if (not sim_isGood_isLoaded) {
    if (sim_isGood_branch != 0) {
      sim_isGood_branch->GetEntry(index);
    } else {
      printf("branch sim_isGood_branch does not exist!\n");
      exit(1);
    }
    sim_isGood_isLoaded = true;
  }
  return *sim_isGood_;
}
const std::vector<float> &LSTEff::pT3_pt() {
  if (not pT3_pt_isLoaded) {
    if (pT3_pt_branch != 0) {
      pT3_pt_branch->GetEntry(index);
    } else {
      printf("branch pT3_pt_branch does not exist!\n");
      exit(1);
    }
    pT3_pt_isLoaded = true;
  }
  return *pT3_pt_;
}
const std::vector<float> &LSTEff::tc_pt() {
  if (not tc_pt_isLoaded) {
    if (tc_pt_branch != 0) {
      tc_pt_branch->GetEntry(index);
    } else {
      printf("branch tc_pt_branch does not exist!\n");
      exit(1);
    }
    tc_pt_isLoaded = true;
  }
  return *tc_pt_;
}
const std::vector<float> &LSTEff::pT3_phi_2() {
  if (not pT3_phi_2_isLoaded) {
    if (pT3_phi_2_branch != 0) {
      pT3_phi_2_branch->GetEntry(index);
    } else {
      printf("branch pT3_phi_2_branch does not exist!\n");
      exit(1);
    }
    pT3_phi_2_isLoaded = true;
  }
  return *pT3_phi_2_;
}
const std::vector<float> &LSTEff::pT5_pt() {
  if (not pT5_pt_isLoaded) {
    if (pT5_pt_branch != 0) {
      pT5_pt_branch->GetEntry(index);
    } else {
      printf("branch pT5_pt_branch does not exist!\n");
      exit(1);
    }
    pT5_pt_isLoaded = true;
  }
  return *pT5_pt_;
}
const std::vector<float> &LSTEff::pureTCE_rPhiChiSquared() {
  if (not pureTCE_rPhiChiSquared_isLoaded) {
    if (pureTCE_rPhiChiSquared_branch != 0) {
      pureTCE_rPhiChiSquared_branch->GetEntry(index);
    } else {
      printf("branch pureTCE_rPhiChiSquared_branch does not exist!\n");
      exit(1);
    }
    pureTCE_rPhiChiSquared_isLoaded = true;
  }
  return *pureTCE_rPhiChiSquared_;
}
const std::vector<int> &LSTEff::pT5_score() {
  if (not pT5_score_isLoaded) {
    if (pT5_score_branch != 0) {
      pT5_score_branch->GetEntry(index);
    } else {
      printf("branch pT5_score_branch does not exist!\n");
      exit(1);
    }
    pT5_score_isLoaded = true;
  }
  return *pT5_score_;
}
const std::vector<float> &LSTEff::sim_phi() {
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
const std::vector<int> &LSTEff::pT5_isFake() {
  if (not pT5_isFake_isLoaded) {
    if (pT5_isFake_branch != 0) {
      pT5_isFake_branch->GetEntry(index);
    } else {
      printf("branch pT5_isFake_branch does not exist!\n");
      exit(1);
    }
    pT5_isFake_isLoaded = true;
  }
  return *pT5_isFake_;
}
const std::vector<int> &LSTEff::tc_maxHitMatchedCounts() {
  if (not tc_maxHitMatchedCounts_isLoaded) {
    if (tc_maxHitMatchedCounts_branch != 0) {
      tc_maxHitMatchedCounts_branch->GetEntry(index);
    } else {
      printf("branch tc_maxHitMatchedCounts_branch does not exist!\n");
      exit(1);
    }
    tc_maxHitMatchedCounts_isLoaded = true;
  }
  return *tc_maxHitMatchedCounts_;
}
const std::vector<std::vector<int> > &LSTEff::pureTCE_nLayerOverlaps() {
  if (not pureTCE_nLayerOverlaps_isLoaded) {
    if (pureTCE_nLayerOverlaps_branch != 0) {
      pureTCE_nLayerOverlaps_branch->GetEntry(index);
    } else {
      printf("branch pureTCE_nLayerOverlaps_branch does not exist!\n");
      exit(1);
    }
    pureTCE_nLayerOverlaps_isLoaded = true;
  }
  return *pureTCE_nLayerOverlaps_;
}
const std::vector<float> &LSTEff::sim_pca_dz() {
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
const std::vector<std::vector<int> > &LSTEff::pureTCE_hitIdxs() {
  if (not pureTCE_hitIdxs_isLoaded) {
    if (pureTCE_hitIdxs_branch != 0) {
      pureTCE_hitIdxs_branch->GetEntry(index);
    } else {
      printf("branch pureTCE_hitIdxs_branch does not exist!\n");
      exit(1);
    }
    pureTCE_hitIdxs_isLoaded = true;
  }
  return *pureTCE_hitIdxs_;
}
const std::vector<std::vector<int> > &LSTEff::pureTCE_nHitOverlaps() {
  if (not pureTCE_nHitOverlaps_isLoaded) {
    if (pureTCE_nHitOverlaps_branch != 0) {
      pureTCE_nHitOverlaps_branch->GetEntry(index);
    } else {
      printf("branch pureTCE_nHitOverlaps_branch does not exist!\n");
      exit(1);
    }
    pureTCE_nHitOverlaps_isLoaded = true;
  }
  return *pureTCE_nHitOverlaps_;
}
const std::vector<int> &LSTEff::sim_pLS_matched() {
  if (not sim_pLS_matched_isLoaded) {
    if (sim_pLS_matched_branch != 0) {
      sim_pLS_matched_branch->GetEntry(index);
    } else {
      printf("branch sim_pLS_matched_branch does not exist!\n");
      exit(1);
    }
    sim_pLS_matched_isLoaded = true;
  }
  return *sim_pLS_matched_;
}
const std::vector<std::vector<int> > &LSTEff::tc_matched_simIdx() {
  if (not tc_matched_simIdx_isLoaded) {
    if (tc_matched_simIdx_branch != 0) {
      tc_matched_simIdx_branch->GetEntry(index);
    } else {
      printf("branch tc_matched_simIdx_branch does not exist!\n");
      exit(1);
    }
    tc_matched_simIdx_isLoaded = true;
  }
  return *tc_matched_simIdx_;
}
const std::vector<int> &LSTEff::sim_T3_matched() {
  if (not sim_T3_matched_isLoaded) {
    if (sim_T3_matched_branch != 0) {
      sim_T3_matched_branch->GetEntry(index);
    } else {
      printf("branch sim_T3_matched_branch does not exist!\n");
      exit(1);
    }
    sim_T3_matched_isLoaded = true;
  }
  return *sim_T3_matched_;
}
const std::vector<float> &LSTEff::pLS_score() {
  if (not pLS_score_isLoaded) {
    if (pLS_score_branch != 0) {
      pLS_score_branch->GetEntry(index);
    } else {
      printf("branch pLS_score_branch does not exist!\n");
      exit(1);
    }
    pLS_score_isLoaded = true;
  }
  return *pLS_score_;
}
const std::vector<float> &LSTEff::pT3_phi() {
  if (not pT3_phi_isLoaded) {
    if (pT3_phi_branch != 0) {
      pT3_phi_branch->GetEntry(index);
    } else {
      printf("branch pT3_phi_branch does not exist!\n");
      exit(1);
    }
    pT3_phi_isLoaded = true;
  }
  return *pT3_phi_;
}
const std::vector<float> &LSTEff::pT5_eta() {
  if (not pT5_eta_isLoaded) {
    if (pT5_eta_branch != 0) {
      pT5_eta_branch->GetEntry(index);
    } else {
      printf("branch pT5_eta_branch does not exist!\n");
      exit(1);
    }
    pT5_eta_isLoaded = true;
  }
  return *pT5_eta_;
}
const std::vector<float> &LSTEff::tc_phi() {
  if (not tc_phi_isLoaded) {
    if (tc_phi_branch != 0) {
      tc_phi_branch->GetEntry(index);
    } else {
      printf("branch tc_phi_branch does not exist!\n");
      exit(1);
    }
    tc_phi_isLoaded = true;
  }
  return *tc_phi_;
}
const std::vector<float> &LSTEff::t4_eta() {
  if (not t4_eta_isLoaded) {
    if (t4_eta_branch != 0) {
      t4_eta_branch->GetEntry(index);
    } else {
      printf("branch t4_eta_branch does not exist!\n");
      exit(1);
    }
    t4_eta_isLoaded = true;
  }
  return *t4_eta_;
}
const std::vector<int> &LSTEff::pLS_isFake() {
  if (not pLS_isFake_isLoaded) {
    if (pLS_isFake_branch != 0) {
      pLS_isFake_branch->GetEntry(index);
    } else {
      printf("branch pLS_isFake_branch does not exist!\n");
      exit(1);
    }
    pLS_isFake_isLoaded = true;
  }
  return *pLS_isFake_;
}
const std::vector<std::vector<int> > &LSTEff::pureTCE_matched_simIdx() {
  if (not pureTCE_matched_simIdx_isLoaded) {
    if (pureTCE_matched_simIdx_branch != 0) {
      pureTCE_matched_simIdx_branch->GetEntry(index);
    } else {
      printf("branch pureTCE_matched_simIdx_branch does not exist!\n");
      exit(1);
    }
    pureTCE_matched_simIdx_isLoaded = true;
  }
  return *pureTCE_matched_simIdx_;
}
const std::vector<int> &LSTEff::sim_bunchCrossing() {
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
const std::vector<int> &LSTEff::tc_partOfExtension() {
  if (not tc_partOfExtension_isLoaded) {
    if (tc_partOfExtension_branch != 0) {
      tc_partOfExtension_branch->GetEntry(index);
    } else {
      printf("branch tc_partOfExtension_branch does not exist!\n");
      exit(1);
    }
    tc_partOfExtension_isLoaded = true;
  }
  return *tc_partOfExtension_;
}
const std::vector<float> &LSTEff::pT3_eta() {
  if (not pT3_eta_isLoaded) {
    if (pT3_eta_branch != 0) {
      pT3_eta_branch->GetEntry(index);
    } else {
      printf("branch pT3_eta_branch does not exist!\n");
      exit(1);
    }
    pT3_eta_isLoaded = true;
  }
  return *pT3_eta_;
}
const std::vector<int> &LSTEff::sim_parentVtxIdx() {
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
const std::vector<int> &LSTEff::pureTCE_layer_binary() {
  if (not pureTCE_layer_binary_isLoaded) {
    if (pureTCE_layer_binary_branch != 0) {
      pureTCE_layer_binary_branch->GetEntry(index);
    } else {
      printf("branch pureTCE_layer_binary_branch does not exist!\n");
      exit(1);
    }
    pureTCE_layer_binary_isLoaded = true;
  }
  return *pureTCE_layer_binary_;
}
const std::vector<int> &LSTEff::sim_pT4_matched() {
  if (not sim_pT4_matched_isLoaded) {
    if (sim_pT4_matched_branch != 0) {
      sim_pT4_matched_branch->GetEntry(index);
    } else {
      printf("branch sim_pT4_matched_branch does not exist!\n");
      exit(1);
    }
    sim_pT4_matched_isLoaded = true;
  }
  return *sim_pT4_matched_;
}
const std::vector<float> &LSTEff::tc_eta() {
  if (not tc_eta_isLoaded) {
    if (tc_eta_branch != 0) {
      tc_eta_branch->GetEntry(index);
    } else {
      printf("branch tc_eta_branch does not exist!\n");
      exit(1);
    }
    tc_eta_isLoaded = true;
  }
  return *tc_eta_;
}
const std::vector<float> &LSTEff::sim_lengap() {
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
const std::vector<int> &LSTEff::sim_T5_matched() {
  if (not sim_T5_matched_isLoaded) {
    if (sim_T5_matched_branch != 0) {
      sim_T5_matched_branch->GetEntry(index);
    } else {
      printf("branch sim_T5_matched_branch does not exist!\n");
      exit(1);
    }
    sim_T5_matched_isLoaded = true;
  }
  return *sim_T5_matched_;
}
const std::vector<std::vector<int> > &LSTEff::sim_T5_types() {
  if (not sim_T5_types_isLoaded) {
    if (sim_T5_types_branch != 0) {
      sim_T5_types_branch->GetEntry(index);
    } else {
      printf("branch sim_T5_types_branch does not exist!\n");
      exit(1);
    }
    sim_T5_types_isLoaded = true;
  }
  return *sim_T5_types_;
}
const std::vector<std::vector<int> > &LSTEff::tce_matched_simIdx() {
  if (not tce_matched_simIdx_isLoaded) {
    if (tce_matched_simIdx_branch != 0) {
      tce_matched_simIdx_branch->GetEntry(index);
    } else {
      printf("branch tce_matched_simIdx_branch does not exist!\n");
      exit(1);
    }
    tce_matched_simIdx_isLoaded = true;
  }
  return *tce_matched_simIdx_;
}
const std::vector<int> &LSTEff::t5_isDuplicate() {
  if (not t5_isDuplicate_isLoaded) {
    if (t5_isDuplicate_branch != 0) {
      t5_isDuplicate_branch->GetEntry(index);
    } else {
      printf("branch t5_isDuplicate_branch does not exist!\n");
      exit(1);
    }
    t5_isDuplicate_isLoaded = true;
  }
  return *t5_isDuplicate_;
}
const std::vector<std::vector<int> > &LSTEff::pT3_hitIdxs() {
  if (not pT3_hitIdxs_isLoaded) {
    if (pT3_hitIdxs_branch != 0) {
      pT3_hitIdxs_branch->GetEntry(index);
    } else {
      printf("branch pT3_hitIdxs_branch does not exist!\n");
      exit(1);
    }
    pT3_hitIdxs_isLoaded = true;
  }
  return *pT3_hitIdxs_;
}
const std::vector<std::vector<int> > &LSTEff::tc_hitIdxs() {
  if (not tc_hitIdxs_isLoaded) {
    if (tc_hitIdxs_branch != 0) {
      tc_hitIdxs_branch->GetEntry(index);
    } else {
      printf("branch tc_hitIdxs_branch does not exist!\n");
      exit(1);
    }
    tc_hitIdxs_isLoaded = true;
  }
  return *tc_hitIdxs_;
}
const int &LSTEff::pT3_occupancies() {
  if (not pT3_occupancies_isLoaded) {
    if (pT3_occupancies_branch != 0) {
      pT3_occupancies_branch->GetEntry(index);
    } else {
      printf("branch pT3_occupancies_branch does not exist!\n");
      exit(1);
    }
    pT3_occupancies_isLoaded = true;
  }
  return pT3_occupancies_;
}
const int &LSTEff::tc_occupancies() {
  if (not tc_occupancies_isLoaded) {
    if (tc_occupancies_branch != 0) {
      tc_occupancies_branch->GetEntry(index);
    } else {
      printf("branch tc_occupancies_branch does not exist!\n");
      exit(1);
    }
    tc_occupancies_isLoaded = true;
  }
  return tc_occupancies_;
}
const std::vector<int> &LSTEff::sim_TC_matched() {
  if (not sim_TC_matched_isLoaded) {
    if (sim_TC_matched_branch != 0) {
      sim_TC_matched_branch->GetEntry(index);
    } else {
      printf("branch sim_TC_matched_branch does not exist!\n");
      exit(1);
    }
    sim_TC_matched_isLoaded = true;
  }
  return *sim_TC_matched_;
}
const std::vector<int> &LSTEff::sim_TC_matched_mask() {
  if (not sim_TC_matched_mask_isLoaded) {
    if (sim_TC_matched_mask_branch != 0) {
      sim_TC_matched_mask_branch->GetEntry(index);
    } else {
      printf("branch sim_TC_matched_mask_branch does not exist!\n");
      exit(1);
    }
    sim_TC_matched_mask_isLoaded = true;
  }
  return *sim_TC_matched_mask_;
}
const std::vector<int> &LSTEff::pLS_isDuplicate() {
  if (not pLS_isDuplicate_isLoaded) {
    if (pLS_isDuplicate_branch != 0) {
      pLS_isDuplicate_branch->GetEntry(index);
    } else {
      printf("branch pLS_isDuplicate_branch does not exist!\n");
      exit(1);
    }
    pLS_isDuplicate_isLoaded = true;
  }
  return *pLS_isDuplicate_;
}
const std::vector<int> &LSTEff::tce_anchorIndex() {
  if (not tce_anchorIndex_isLoaded) {
    if (tce_anchorIndex_branch != 0) {
      tce_anchorIndex_branch->GetEntry(index);
    } else {
      printf("branch tce_anchorIndex_branch does not exist!\n");
      exit(1);
    }
    tce_anchorIndex_isLoaded = true;
  }
  return *tce_anchorIndex_;
}
const std::vector<int> &LSTEff::t5_occupancies() {
  if (not t5_occupancies_isLoaded) {
    if (t5_occupancies_branch != 0) {
      t5_occupancies_branch->GetEntry(index);
    } else {
      printf("branch t5_occupancies_branch does not exist!\n");
      exit(1);
    }
    t5_occupancies_isLoaded = true;
  }
  return *t5_occupancies_;
}
const std::vector<int> &LSTEff::tc_type() {
  if (not tc_type_isLoaded) {
    if (tc_type_branch != 0) {
      tc_type_branch->GetEntry(index);
    } else {
      printf("branch tc_type_branch does not exist!\n");
      exit(1);
    }
    tc_type_isLoaded = true;
  }
  return *tc_type_;
}
const std::vector<int> &LSTEff::tce_isFake() {
  if (not tce_isFake_isLoaded) {
    if (tce_isFake_branch != 0) {
      tce_isFake_branch->GetEntry(index);
    } else {
      printf("branch tce_isFake_branch does not exist!\n");
      exit(1);
    }
    tce_isFake_isLoaded = true;
  }
  return *tce_isFake_;
}
const std::vector<float> &LSTEff::pLS_pt() {
  if (not pLS_pt_isLoaded) {
    if (pLS_pt_branch != 0) {
      pLS_pt_branch->GetEntry(index);
    } else {
      printf("branch pLS_pt_branch does not exist!\n");
      exit(1);
    }
    pLS_pt_isLoaded = true;
  }
  return *pLS_pt_;
}
const std::vector<int> &LSTEff::pureTCE_anchorIndex() {
  if (not pureTCE_anchorIndex_isLoaded) {
    if (pureTCE_anchorIndex_branch != 0) {
      pureTCE_anchorIndex_branch->GetEntry(index);
    } else {
      printf("branch pureTCE_anchorIndex_branch does not exist!\n");
      exit(1);
    }
    pureTCE_anchorIndex_isLoaded = true;
  }
  return *pureTCE_anchorIndex_;
}
const std::vector<std::vector<int> > &LSTEff::sim_T4_types() {
  if (not sim_T4_types_isLoaded) {
    if (sim_T4_types_branch != 0) {
      sim_T4_types_branch->GetEntry(index);
    } else {
      printf("branch sim_T4_types_branch does not exist!\n");
      exit(1);
    }
    sim_T4_types_isLoaded = true;
  }
  return *sim_T4_types_;
}
const std::vector<int> &LSTEff::pT4_isDuplicate() {
  if (not pT4_isDuplicate_isLoaded) {
    if (pT4_isDuplicate_branch != 0) {
      pT4_isDuplicate_branch->GetEntry(index);
    } else {
      printf("branch pT4_isDuplicate_branch does not exist!\n");
      exit(1);
    }
    pT4_isDuplicate_isLoaded = true;
  }
  return *pT4_isDuplicate_;
}
const std::vector<float> &LSTEff::t4_pt() {
  if (not t4_pt_isLoaded) {
    if (t4_pt_branch != 0) {
      t4_pt_branch->GetEntry(index);
    } else {
      printf("branch t4_pt_branch does not exist!\n");
      exit(1);
    }
    t4_pt_isLoaded = true;
  }
  return *t4_pt_;
}
const std::vector<std::vector<int> > &LSTEff::sim_TC_types() {
  if (not sim_TC_types_isLoaded) {
    if (sim_TC_types_branch != 0) {
      sim_TC_types_branch->GetEntry(index);
    } else {
      printf("branch sim_TC_types_branch does not exist!\n");
      exit(1);
    }
    sim_TC_types_isLoaded = true;
  }
  return *sim_TC_types_;
}
const std::vector<int> &LSTEff::sg_occupancies() {
  if (not sg_occupancies_isLoaded) {
    if (sg_occupancies_branch != 0) {
      sg_occupancies_branch->GetEntry(index);
    } else {
      printf("branch sg_occupancies_branch does not exist!\n");
      exit(1);
    }
    sg_occupancies_isLoaded = true;
  }
  return *sg_occupancies_;
}
const std::vector<float> &LSTEff::pT4_pt() {
  if (not pT4_pt_isLoaded) {
    if (pT4_pt_branch != 0) {
      pT4_pt_branch->GetEntry(index);
    } else {
      printf("branch pT4_pt_branch does not exist!\n");
      exit(1);
    }
    pT4_pt_isLoaded = true;
  }
  return *pT4_pt_;
}
const std::vector<float> &LSTEff::pureTCE_phi() {
  if (not pureTCE_phi_isLoaded) {
    if (pureTCE_phi_branch != 0) {
      pureTCE_phi_branch->GetEntry(index);
    } else {
      printf("branch pureTCE_phi_branch does not exist!\n");
      exit(1);
    }
    pureTCE_phi_isLoaded = true;
  }
  return *pureTCE_phi_;
}
const std::vector<float> &LSTEff::sim_vx() {
  if (not sim_vx_isLoaded) {
    if (sim_vx_branch != 0) {
      sim_vx_branch->GetEntry(index);
    } else {
      printf("branch sim_vx_branch does not exist!\n");
      exit(1);
    }
    sim_vx_isLoaded = true;
  }
  return *sim_vx_;
}
const std::vector<float> &LSTEff::sim_vy() {
  if (not sim_vy_isLoaded) {
    if (sim_vy_branch != 0) {
      sim_vy_branch->GetEntry(index);
    } else {
      printf("branch sim_vy_branch does not exist!\n");
      exit(1);
    }
    sim_vy_isLoaded = true;
  }
  return *sim_vy_;
}
const std::vector<float> &LSTEff::sim_vz() {
  if (not sim_vz_isLoaded) {
    if (sim_vz_branch != 0) {
      sim_vz_branch->GetEntry(index);
    } else {
      printf("branch sim_vz_branch does not exist!\n");
      exit(1);
    }
    sim_vz_isLoaded = true;
  }
  return *sim_vz_;
}
const std::vector<int> &LSTEff::tce_maxHitMatchedCounts() {
  if (not tce_maxHitMatchedCounts_isLoaded) {
    if (tce_maxHitMatchedCounts_branch != 0) {
      tce_maxHitMatchedCounts_branch->GetEntry(index);
    } else {
      printf("branch tce_maxHitMatchedCounts_branch does not exist!\n");
      exit(1);
    }
    tce_maxHitMatchedCounts_isLoaded = true;
  }
  return *tce_maxHitMatchedCounts_;
}
const std::vector<float> &LSTEff::t3_pt() {
  if (not t3_pt_isLoaded) {
    if (t3_pt_branch != 0) {
      t3_pt_branch->GetEntry(index);
    } else {
      printf("branch t3_pt_branch does not exist!\n");
      exit(1);
    }
    t3_pt_isLoaded = true;
  }
  return *t3_pt_;
}
const std::vector<int> &LSTEff::module_rings() {
  if (not module_rings_isLoaded) {
    if (module_rings_branch != 0) {
      module_rings_branch->GetEntry(index);
    } else {
      printf("branch module_rings_branch does not exist!\n");
      exit(1);
    }
    module_rings_isLoaded = true;
  }
  return *module_rings_;
}
const std::vector<std::vector<int> > &LSTEff::sim_T3_types() {
  if (not sim_T3_types_isLoaded) {
    if (sim_T3_types_branch != 0) {
      sim_T3_types_branch->GetEntry(index);
    } else {
      printf("branch sim_T3_types_branch does not exist!\n");
      exit(1);
    }
    sim_T3_types_isLoaded = true;
  }
  return *sim_T3_types_;
}
const std::vector<std::vector<int> > &LSTEff::sim_pT5_types() {
  if (not sim_pT5_types_isLoaded) {
    if (sim_pT5_types_branch != 0) {
      sim_pT5_types_branch->GetEntry(index);
    } else {
      printf("branch sim_pT5_types_branch does not exist!\n");
      exit(1);
    }
    sim_pT5_types_isLoaded = true;
  }
  return *sim_pT5_types_;
}
const std::vector<int> &LSTEff::sim_pT5_matched() {
  if (not sim_pT5_matched_isLoaded) {
    if (sim_pT5_matched_branch != 0) {
      sim_pT5_matched_branch->GetEntry(index);
    } else {
      printf("branch sim_pT5_matched_branch does not exist!\n");
      exit(1);
    }
    sim_pT5_matched_isLoaded = true;
  }
  return *sim_pT5_matched_;
}
const std::vector<int> &LSTEff::module_layers() {
  if (not module_layers_isLoaded) {
    if (module_layers_branch != 0) {
      module_layers_branch->GetEntry(index);
    } else {
      printf("branch module_layers_branch does not exist!\n");
      exit(1);
    }
    module_layers_isLoaded = true;
  }
  return *module_layers_;
}
const std::vector<float> &LSTEff::pT4_eta() {
  if (not pT4_eta_isLoaded) {
    if (pT4_eta_branch != 0) {
      pT4_eta_branch->GetEntry(index);
    } else {
      printf("branch pT4_eta_branch does not exist!\n");
      exit(1);
    }
    pT4_eta_isLoaded = true;
  }
  return *pT4_eta_;
}
const std::vector<std::vector<int> > &LSTEff::sim_tce_types() {
  if (not sim_tce_types_isLoaded) {
    if (sim_tce_types_branch != 0) {
      sim_tce_types_branch->GetEntry(index);
    } else {
      printf("branch sim_tce_types_branch does not exist!\n");
      exit(1);
    }
    sim_tce_types_isLoaded = true;
  }
  return *sim_tce_types_;
}
const std::vector<float> &LSTEff::tce_rzChiSquared() {
  if (not tce_rzChiSquared_isLoaded) {
    if (tce_rzChiSquared_branch != 0) {
      tce_rzChiSquared_branch->GetEntry(index);
    } else {
      printf("branch tce_rzChiSquared_branch does not exist!\n");
      exit(1);
    }
    tce_rzChiSquared_isLoaded = true;
  }
  return *tce_rzChiSquared_;
}
const std::vector<std::vector<int> > &LSTEff::pT3_matched_simIdx() {
  if (not pT3_matched_simIdx_isLoaded) {
    if (pT3_matched_simIdx_branch != 0) {
      pT3_matched_simIdx_branch->GetEntry(index);
    } else {
      printf("branch pT3_matched_simIdx_branch does not exist!\n");
      exit(1);
    }
    pT3_matched_simIdx_isLoaded = true;
  }
  return *pT3_matched_simIdx_;
}
void LSTEff::progress(int nEventsTotal, int nEventsChain) {
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
  const int &pT5_occupancies() { return lstEff.pT5_occupancies(); }
  const std::vector<float> &t3_phi() { return lstEff.t3_phi(); }
  const std::vector<float> &t5_score_rphisum() { return lstEff.t5_score_rphisum(); }
  const std::vector<int> &pT4_isFake() { return lstEff.pT4_isFake(); }
  const std::vector<int> &t3_isDuplicate() { return lstEff.t3_isDuplicate(); }
  const std::vector<int> &sim_event() { return lstEff.sim_event(); }
  const std::vector<int> &sim_q() { return lstEff.sim_q(); }
  const std::vector<float> &sim_eta() { return lstEff.sim_eta(); }
  const std::vector<int> &pT3_foundDuplicate() { return lstEff.pT3_foundDuplicate(); }
  const std::vector<float> &sim_len() { return lstEff.sim_len(); }
  const std::vector<int> &pureTCE_isDuplicate() { return lstEff.pureTCE_isDuplicate(); }
  const std::vector<float> &pT3_score() { return lstEff.pT3_score(); }
  const std::vector<float> &t5_eta() { return lstEff.t5_eta(); }
  const std::vector<int> &sim_denom() { return lstEff.sim_denom(); }
  const std::vector<int> &pT5_isDuplicate() { return lstEff.pT5_isDuplicate(); }
  const std::vector<int> &sim_tce_matched() { return lstEff.sim_tce_matched(); }
  const std::vector<int> &pT3_isDuplicate() { return lstEff.pT3_isDuplicate(); }
  const std::vector<int> &tc_isDuplicate() { return lstEff.tc_isDuplicate(); }
  const std::vector<float> &pT3_eta_2() { return lstEff.pT3_eta_2(); }
  const std::vector<int> &sim_pT3_matched() { return lstEff.sim_pT3_matched(); }
  const std::vector<float> &pureTCE_rzChiSquared() { return lstEff.pureTCE_rzChiSquared(); }
  const std::vector<int> &t4_isDuplicate() { return lstEff.t4_isDuplicate(); }
  const std::vector<float> &pureTCE_eta() { return lstEff.pureTCE_eta(); }
  const std::vector<float> &tce_rPhiChiSquared() { return lstEff.tce_rPhiChiSquared(); }
  const std::vector<int> &pureTCE_anchorType() { return lstEff.pureTCE_anchorType(); }
  const std::vector<float> &pureTCE_pt() { return lstEff.pureTCE_pt(); }
  const std::vector<float> &sim_pt() { return lstEff.sim_pt(); }
  const std::vector<float> &t5_eta_2() { return lstEff.t5_eta_2(); }
  const std::vector<float> &pLS_eta() { return lstEff.pLS_eta(); }
  const std::vector<int> &sim_pdgId() { return lstEff.sim_pdgId(); }
  const std::vector<float> &t3_eta() { return lstEff.t3_eta(); }
  const std::vector<int> &tce_layer_binary() { return lstEff.tce_layer_binary(); }
  const std::vector<int> &sim_TC_matched_nonextended() { return lstEff.sim_TC_matched_nonextended(); }
  const std::vector<int> &t4_occupancies() { return lstEff.t4_occupancies(); }
  const std::vector<float> &tce_eta() { return lstEff.tce_eta(); }
  const std::vector<int> &tce_isDuplicate() { return lstEff.tce_isDuplicate(); }
  const std::vector<std::vector<int> > &pT5_matched_simIdx() { return lstEff.pT5_matched_simIdx(); }
  const std::vector<std::vector<int> > &sim_tcIdx() { return lstEff.sim_tcIdx(); }
  const std::vector<float> &t5_phi_2() { return lstEff.t5_phi_2(); }
  const std::vector<int> &pureTCE_maxHitMatchedCounts() { return lstEff.pureTCE_maxHitMatchedCounts(); }
  const std::vector<std::vector<int> > &t5_matched_simIdx() { return lstEff.t5_matched_simIdx(); }
  const std::vector<int> &module_subdets() { return lstEff.module_subdets(); }
  const std::vector<int> &tce_anchorType() { return lstEff.tce_anchorType(); }
  const std::vector<std::vector<int> > &tce_nHitOverlaps() { return lstEff.tce_nHitOverlaps(); }
  const std::vector<int> &t3_isFake() { return lstEff.t3_isFake(); }
  const std::vector<float> &tce_phi() { return lstEff.tce_phi(); }
  const std::vector<int> &t5_isFake() { return lstEff.t5_isFake(); }
  const std::vector<int> &md_occupancies() { return lstEff.md_occupancies(); }
  const std::vector<std::vector<int> > &t5_hitIdxs() { return lstEff.t5_hitIdxs(); }
  const std::vector<std::vector<int> > &sim_pT3_types() { return lstEff.sim_pT3_types(); }
  const std::vector<std::vector<int> > &sim_pureTCE_types() { return lstEff.sim_pureTCE_types(); }
  const std::vector<float> &t4_phi() { return lstEff.t4_phi(); }
  const std::vector<float> &t5_phi() { return lstEff.t5_phi(); }
  const std::vector<std::vector<int> > &pT5_hitIdxs() { return lstEff.pT5_hitIdxs(); }
  const std::vector<float> &t5_pt() { return lstEff.t5_pt(); }
  const std::vector<float> &pT5_phi() { return lstEff.pT5_phi(); }
  const std::vector<int> &pureTCE_isFake() { return lstEff.pureTCE_isFake(); }
  const std::vector<float> &tce_pt() { return lstEff.tce_pt(); }
  const std::vector<int> &tc_isFake() { return lstEff.tc_isFake(); }
  const std::vector<int> &pT3_isFake() { return lstEff.pT3_isFake(); }
  const std::vector<std::vector<int> > &tce_nLayerOverlaps() { return lstEff.tce_nLayerOverlaps(); }
  const std::vector<int> &tc_sim() { return lstEff.tc_sim(); }
  const std::vector<std::vector<int> > &sim_pLS_types() { return lstEff.sim_pLS_types(); }
  const std::vector<float> &sim_pca_dxy() { return lstEff.sim_pca_dxy(); }
  const std::vector<float> &pT4_phi() { return lstEff.pT4_phi(); }
  const std::vector<float> &sim_hits() { return lstEff.sim_hits(); }
  const std::vector<float> &pLS_phi() { return lstEff.pLS_phi(); }
  const std::vector<int> &sim_pureTCE_matched() { return lstEff.sim_pureTCE_matched(); }
  const std::vector<int> &t3_occupancies() { return lstEff.t3_occupancies(); }
  const std::vector<int> &t5_foundDuplicate() { return lstEff.t5_foundDuplicate(); }
  const std::vector<std::vector<int> > &sim_pT4_types() { return lstEff.sim_pT4_types(); }
  const std::vector<int> &t4_isFake() { return lstEff.t4_isFake(); }
  const std::vector<float> &simvtx_x() { return lstEff.simvtx_x(); }
  const std::vector<float> &simvtx_y() { return lstEff.simvtx_y(); }
  const std::vector<float> &simvtx_z() { return lstEff.simvtx_z(); }
  const std::vector<int> &sim_T4_matched() { return lstEff.sim_T4_matched(); }
  const std::vector<bool> &sim_isGood() { return lstEff.sim_isGood(); }
  const std::vector<float> &pT3_pt() { return lstEff.pT3_pt(); }
  const std::vector<float> &tc_pt() { return lstEff.tc_pt(); }
  const std::vector<float> &pT3_phi_2() { return lstEff.pT3_phi_2(); }
  const std::vector<float> &pT5_pt() { return lstEff.pT5_pt(); }
  const std::vector<float> &pureTCE_rPhiChiSquared() { return lstEff.pureTCE_rPhiChiSquared(); }
  const std::vector<int> &pT5_score() { return lstEff.pT5_score(); }
  const std::vector<float> &sim_phi() { return lstEff.sim_phi(); }
  const std::vector<int> &pT5_isFake() { return lstEff.pT5_isFake(); }
  const std::vector<int> &tc_maxHitMatchedCounts() { return lstEff.tc_maxHitMatchedCounts(); }
  const std::vector<std::vector<int> > &pureTCE_nLayerOverlaps() { return lstEff.pureTCE_nLayerOverlaps(); }
  const std::vector<float> &sim_pca_dz() { return lstEff.sim_pca_dz(); }
  const std::vector<std::vector<int> > &pureTCE_hitIdxs() { return lstEff.pureTCE_hitIdxs(); }
  const std::vector<std::vector<int> > &pureTCE_nHitOverlaps() { return lstEff.pureTCE_nHitOverlaps(); }
  const std::vector<int> &sim_pLS_matched() { return lstEff.sim_pLS_matched(); }
  const std::vector<std::vector<int> > &tc_matched_simIdx() { return lstEff.tc_matched_simIdx(); }
  const std::vector<int> &sim_T3_matched() { return lstEff.sim_T3_matched(); }
  const std::vector<float> &pLS_score() { return lstEff.pLS_score(); }
  const std::vector<float> &pT3_phi() { return lstEff.pT3_phi(); }
  const std::vector<float> &pT5_eta() { return lstEff.pT5_eta(); }
  const std::vector<float> &tc_phi() { return lstEff.tc_phi(); }
  const std::vector<float> &t4_eta() { return lstEff.t4_eta(); }
  const std::vector<int> &pLS_isFake() { return lstEff.pLS_isFake(); }
  const std::vector<std::vector<int> > &pureTCE_matched_simIdx() { return lstEff.pureTCE_matched_simIdx(); }
  const std::vector<int> &sim_bunchCrossing() { return lstEff.sim_bunchCrossing(); }
  const std::vector<int> &tc_partOfExtension() { return lstEff.tc_partOfExtension(); }
  const std::vector<float> &pT3_eta() { return lstEff.pT3_eta(); }
  const std::vector<int> &sim_parentVtxIdx() { return lstEff.sim_parentVtxIdx(); }
  const std::vector<int> &pureTCE_layer_binary() { return lstEff.pureTCE_layer_binary(); }
  const std::vector<int> &sim_pT4_matched() { return lstEff.sim_pT4_matched(); }
  const std::vector<float> &tc_eta() { return lstEff.tc_eta(); }
  const std::vector<float> &sim_lengap() { return lstEff.sim_lengap(); }
  const std::vector<int> &sim_T5_matched() { return lstEff.sim_T5_matched(); }
  const std::vector<std::vector<int> > &sim_T5_types() { return lstEff.sim_T5_types(); }
  const std::vector<std::vector<int> > &tce_matched_simIdx() { return lstEff.tce_matched_simIdx(); }
  const std::vector<int> &t5_isDuplicate() { return lstEff.t5_isDuplicate(); }
  const std::vector<std::vector<int> > &pT3_hitIdxs() { return lstEff.pT3_hitIdxs(); }
  const std::vector<std::vector<int> > &tc_hitIdxs() { return lstEff.tc_hitIdxs(); }
  const int &pT3_occupancies() { return lstEff.pT3_occupancies(); }
  const int &tc_occupancies() { return lstEff.tc_occupancies(); }
  const std::vector<int> &sim_TC_matched() { return lstEff.sim_TC_matched(); }
  const std::vector<int> &sim_TC_matched_mask() { return lstEff.sim_TC_matched_mask(); }
  const std::vector<int> &pLS_isDuplicate() { return lstEff.pLS_isDuplicate(); }
  const std::vector<int> &tce_anchorIndex() { return lstEff.tce_anchorIndex(); }
  const std::vector<int> &t5_occupancies() { return lstEff.t5_occupancies(); }
  const std::vector<int> &tc_type() { return lstEff.tc_type(); }
  const std::vector<int> &tce_isFake() { return lstEff.tce_isFake(); }
  const std::vector<float> &pLS_pt() { return lstEff.pLS_pt(); }
  const std::vector<int> &pureTCE_anchorIndex() { return lstEff.pureTCE_anchorIndex(); }
  const std::vector<std::vector<int> > &sim_T4_types() { return lstEff.sim_T4_types(); }
  const std::vector<int> &pT4_isDuplicate() { return lstEff.pT4_isDuplicate(); }
  const std::vector<float> &t4_pt() { return lstEff.t4_pt(); }
  const std::vector<std::vector<int> > &sim_TC_types() { return lstEff.sim_TC_types(); }
  const std::vector<int> &sg_occupancies() { return lstEff.sg_occupancies(); }
  const std::vector<float> &pT4_pt() { return lstEff.pT4_pt(); }
  const std::vector<float> &pureTCE_phi() { return lstEff.pureTCE_phi(); }
  const std::vector<float> &sim_vx() { return lstEff.sim_vx(); }
  const std::vector<float> &sim_vy() { return lstEff.sim_vy(); }
  const std::vector<float> &sim_vz() { return lstEff.sim_vz(); }
  const std::vector<int> &tce_maxHitMatchedCounts() { return lstEff.tce_maxHitMatchedCounts(); }
  const std::vector<float> &t3_pt() { return lstEff.t3_pt(); }
  const std::vector<int> &module_rings() { return lstEff.module_rings(); }
  const std::vector<std::vector<int> > &sim_T3_types() { return lstEff.sim_T3_types(); }
  const std::vector<std::vector<int> > &sim_pT5_types() { return lstEff.sim_pT5_types(); }
  const std::vector<int> &sim_pT5_matched() { return lstEff.sim_pT5_matched(); }
  const std::vector<int> &module_layers() { return lstEff.module_layers(); }
  const std::vector<float> &pT4_eta() { return lstEff.pT4_eta(); }
  const std::vector<std::vector<int> > &sim_tce_types() { return lstEff.sim_tce_types(); }
  const std::vector<float> &tce_rzChiSquared() { return lstEff.tce_rzChiSquared(); }
  const std::vector<std::vector<int> > &pT3_matched_simIdx() { return lstEff.pT3_matched_simIdx(); }
}  // namespace tas
