# /dev/CMSSW_15_0_0/GRun

import FWCore.ParameterSet.Config as cms


# stream ParkingAnomalyDetection

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingAnomalyDetection_datasetParkingAnomalyDetection_selector
streamParkingAnomalyDetection_datasetParkingAnomalyDetection_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingAnomalyDetection_datasetParkingAnomalyDetection_selector.l1tResults = cms.InputTag('')
streamParkingAnomalyDetection_datasetParkingAnomalyDetection_selector.throw      = cms.bool(False)
streamParkingAnomalyDetection_datasetParkingAnomalyDetection_selector.triggerConditions = cms.vstring(
    'HLT_L1AXOVVTight_v1',
    'HLT_L1AXOVVVTight_v1',
    'HLT_L1CICADA_VVTight_v1',
    'HLT_L1CICADA_VVVTight_v1',
    'HLT_L1CICADA_VVVVTight_v1'
)


# stream ParkingDoubleMuonLowMass0

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingDoubleMuonLowMass0_datasetParkingDoubleMuonLowMass0_selector
streamParkingDoubleMuonLowMass0_datasetParkingDoubleMuonLowMass0_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingDoubleMuonLowMass0_datasetParkingDoubleMuonLowMass0_selector.l1tResults = cms.InputTag('')
streamParkingDoubleMuonLowMass0_datasetParkingDoubleMuonLowMass0_selector.throw      = cms.bool(False)
streamParkingDoubleMuonLowMass0_datasetParkingDoubleMuonLowMass0_selector.triggerConditions = cms.vstring(
    'HLT_Dimuon0_Jpsi3p5_Muon2_v18',
    'HLT_Dimuon0_Jpsi_L1_4R_0er1p5R_v20',
    'HLT_Dimuon0_Jpsi_L1_NoOS_v20',
    'HLT_Dimuon0_Jpsi_NoVertexing_L1_4R_0er1p5R_v20',
    'HLT_Dimuon0_Jpsi_NoVertexing_NoOS_v20',
    'HLT_Dimuon0_Jpsi_NoVertexing_v21',
    'HLT_Dimuon0_Jpsi_v21',
    'HLT_Dimuon0_LowMass_L1_0er1p5_v21',
    'HLT_Dimuon0_LowMass_L1_4_v21',
    'HLT_Dimuon0_LowMass_L1_TM530_v19',
    'HLT_Dimuon0_LowMass_v21',
    'HLT_Dimuon0_Upsilon_L1_4p5_v22',
    'HLT_Dimuon0_Upsilon_L1_4p5er2p0M_v20',
    'HLT_Dimuon0_Upsilon_L1_4p5er2p0_v22',
    'HLT_Dimuon0_Upsilon_Muon_NoL1Mass_v19',
    'HLT_Dimuon0_Upsilon_NoVertexing_v20',
    'HLT_Dimuon10_Upsilon_y1p4_v14',
    'HLT_Dimuon12_Upsilon_y1p4_v15',
    'HLT_Dimuon14_Phi_Barrel_Seagulls_v20',
    'HLT_Dimuon14_PsiPrime_noCorrL1_v18',
    'HLT_Dimuon14_PsiPrime_v26',
    'HLT_Dimuon18_PsiPrime_noCorrL1_v19',
    'HLT_Dimuon18_PsiPrime_v27',
    'HLT_Dimuon24_Phi_noCorrL1_v19',
    'HLT_Dimuon24_Upsilon_noCorrL1_v19',
    'HLT_Dimuon25_Jpsi_noCorrL1_v19',
    'HLT_Dimuon25_Jpsi_v27',
    'HLT_DoubleMu2_Jpsi_DoubleTrk1_Phi1p05_v19',
    'HLT_DoubleMu2_Jpsi_LowPt_v7',
    'HLT_DoubleMu3_DoubleEle7p5_CaloIdL_TrackIdL_Upsilon_v17',
    'HLT_DoubleMu3_TkMu_DsTau3Mu_v17',
    'HLT_DoubleMu3_Trk_Tau3mu_NoL1Mass_v19',
    'HLT_DoubleMu3_Trk_Tau3mu_v25',
    'HLT_DoubleMu4_3_Bs_v28',
    'HLT_DoubleMu4_3_Displaced_Photon4_BsToMMG_v13',
    'HLT_DoubleMu4_3_Jpsi_v28',
    'HLT_DoubleMu4_3_LowMass_SS_v7',
    'HLT_DoubleMu4_3_LowMass_v14',
    'HLT_DoubleMu4_3_Photon4_BsToMMG_v13',
    'HLT_DoubleMu4_JpsiTrkTrk_Displaced_v20',
    'HLT_DoubleMu4_JpsiTrk_Bc_v13',
    'HLT_DoubleMu4_Jpsi_Displaced_v20',
    'HLT_DoubleMu4_Jpsi_NoVertexing_v20',
    'HLT_DoubleMu4_LowMass_Displaced_v14',
    'HLT_DoubleMu4_MuMuTrk_Displaced_v28',
    'HLT_DoubleMu5_Upsilon_DoubleEle3_CaloIdL_TrackIdL_v17',
    'HLT_Mu25_TkMu0_Phi_v21',
    'HLT_Mu30_TkMu0_Psi_v14',
    'HLT_Mu30_TkMu0_Upsilon_v14',
    'HLT_Mu4_L1DoubleMu_v14',
    'HLT_Mu7p5_L2Mu2_Jpsi_v23',
    'HLT_Mu7p5_L2Mu2_Upsilon_v23',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_Charge1_v17',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_v17',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15_Charge1_v17',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15_v17',
    'HLT_Trimuon5_3p5_2_Upsilon_Muon_v18',
    'HLT_TrimuonOpen_5_3p5_2_Upsilon_Muon_v15'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingDoubleMuonLowMass0_datasetParkingDoubleMuonLowMass1_selector
streamParkingDoubleMuonLowMass0_datasetParkingDoubleMuonLowMass1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingDoubleMuonLowMass0_datasetParkingDoubleMuonLowMass1_selector.l1tResults = cms.InputTag('')
streamParkingDoubleMuonLowMass0_datasetParkingDoubleMuonLowMass1_selector.throw      = cms.bool(False)
streamParkingDoubleMuonLowMass0_datasetParkingDoubleMuonLowMass1_selector.triggerConditions = cms.vstring(
    'HLT_Dimuon0_Jpsi3p5_Muon2_v18',
    'HLT_Dimuon0_Jpsi_L1_4R_0er1p5R_v20',
    'HLT_Dimuon0_Jpsi_L1_NoOS_v20',
    'HLT_Dimuon0_Jpsi_NoVertexing_L1_4R_0er1p5R_v20',
    'HLT_Dimuon0_Jpsi_NoVertexing_NoOS_v20',
    'HLT_Dimuon0_Jpsi_NoVertexing_v21',
    'HLT_Dimuon0_Jpsi_v21',
    'HLT_Dimuon0_LowMass_L1_0er1p5_v21',
    'HLT_Dimuon0_LowMass_L1_4_v21',
    'HLT_Dimuon0_LowMass_L1_TM530_v19',
    'HLT_Dimuon0_LowMass_v21',
    'HLT_Dimuon0_Upsilon_L1_4p5_v22',
    'HLT_Dimuon0_Upsilon_L1_4p5er2p0M_v20',
    'HLT_Dimuon0_Upsilon_L1_4p5er2p0_v22',
    'HLT_Dimuon0_Upsilon_Muon_NoL1Mass_v19',
    'HLT_Dimuon0_Upsilon_NoVertexing_v20',
    'HLT_Dimuon10_Upsilon_y1p4_v14',
    'HLT_Dimuon12_Upsilon_y1p4_v15',
    'HLT_Dimuon14_Phi_Barrel_Seagulls_v20',
    'HLT_Dimuon14_PsiPrime_noCorrL1_v18',
    'HLT_Dimuon14_PsiPrime_v26',
    'HLT_Dimuon18_PsiPrime_noCorrL1_v19',
    'HLT_Dimuon18_PsiPrime_v27',
    'HLT_Dimuon24_Phi_noCorrL1_v19',
    'HLT_Dimuon24_Upsilon_noCorrL1_v19',
    'HLT_Dimuon25_Jpsi_noCorrL1_v19',
    'HLT_Dimuon25_Jpsi_v27',
    'HLT_DoubleMu2_Jpsi_DoubleTrk1_Phi1p05_v19',
    'HLT_DoubleMu2_Jpsi_LowPt_v7',
    'HLT_DoubleMu3_DoubleEle7p5_CaloIdL_TrackIdL_Upsilon_v17',
    'HLT_DoubleMu3_TkMu_DsTau3Mu_v17',
    'HLT_DoubleMu3_Trk_Tau3mu_NoL1Mass_v19',
    'HLT_DoubleMu3_Trk_Tau3mu_v25',
    'HLT_DoubleMu4_3_Bs_v28',
    'HLT_DoubleMu4_3_Displaced_Photon4_BsToMMG_v13',
    'HLT_DoubleMu4_3_Jpsi_v28',
    'HLT_DoubleMu4_3_LowMass_SS_v7',
    'HLT_DoubleMu4_3_LowMass_v14',
    'HLT_DoubleMu4_3_Photon4_BsToMMG_v13',
    'HLT_DoubleMu4_JpsiTrkTrk_Displaced_v20',
    'HLT_DoubleMu4_JpsiTrk_Bc_v13',
    'HLT_DoubleMu4_Jpsi_Displaced_v20',
    'HLT_DoubleMu4_Jpsi_NoVertexing_v20',
    'HLT_DoubleMu4_LowMass_Displaced_v14',
    'HLT_DoubleMu4_MuMuTrk_Displaced_v28',
    'HLT_DoubleMu5_Upsilon_DoubleEle3_CaloIdL_TrackIdL_v17',
    'HLT_Mu25_TkMu0_Phi_v21',
    'HLT_Mu30_TkMu0_Psi_v14',
    'HLT_Mu30_TkMu0_Upsilon_v14',
    'HLT_Mu4_L1DoubleMu_v14',
    'HLT_Mu7p5_L2Mu2_Jpsi_v23',
    'HLT_Mu7p5_L2Mu2_Upsilon_v23',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_Charge1_v17',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_v17',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15_Charge1_v17',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15_v17',
    'HLT_Trimuon5_3p5_2_Upsilon_Muon_v18',
    'HLT_TrimuonOpen_5_3p5_2_Upsilon_Muon_v15'
)


# stream ParkingDoubleMuonLowMass1

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingDoubleMuonLowMass1_datasetParkingDoubleMuonLowMass2_selector
streamParkingDoubleMuonLowMass1_datasetParkingDoubleMuonLowMass2_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingDoubleMuonLowMass1_datasetParkingDoubleMuonLowMass2_selector.l1tResults = cms.InputTag('')
streamParkingDoubleMuonLowMass1_datasetParkingDoubleMuonLowMass2_selector.throw      = cms.bool(False)
streamParkingDoubleMuonLowMass1_datasetParkingDoubleMuonLowMass2_selector.triggerConditions = cms.vstring(
    'HLT_Dimuon0_Jpsi3p5_Muon2_v18',
    'HLT_Dimuon0_Jpsi_L1_4R_0er1p5R_v20',
    'HLT_Dimuon0_Jpsi_L1_NoOS_v20',
    'HLT_Dimuon0_Jpsi_NoVertexing_L1_4R_0er1p5R_v20',
    'HLT_Dimuon0_Jpsi_NoVertexing_NoOS_v20',
    'HLT_Dimuon0_Jpsi_NoVertexing_v21',
    'HLT_Dimuon0_Jpsi_v21',
    'HLT_Dimuon0_LowMass_L1_0er1p5_v21',
    'HLT_Dimuon0_LowMass_L1_4_v21',
    'HLT_Dimuon0_LowMass_L1_TM530_v19',
    'HLT_Dimuon0_LowMass_v21',
    'HLT_Dimuon0_Upsilon_L1_4p5_v22',
    'HLT_Dimuon0_Upsilon_L1_4p5er2p0M_v20',
    'HLT_Dimuon0_Upsilon_L1_4p5er2p0_v22',
    'HLT_Dimuon0_Upsilon_Muon_NoL1Mass_v19',
    'HLT_Dimuon0_Upsilon_NoVertexing_v20',
    'HLT_Dimuon10_Upsilon_y1p4_v14',
    'HLT_Dimuon12_Upsilon_y1p4_v15',
    'HLT_Dimuon14_Phi_Barrel_Seagulls_v20',
    'HLT_Dimuon14_PsiPrime_noCorrL1_v18',
    'HLT_Dimuon14_PsiPrime_v26',
    'HLT_Dimuon18_PsiPrime_noCorrL1_v19',
    'HLT_Dimuon18_PsiPrime_v27',
    'HLT_Dimuon24_Phi_noCorrL1_v19',
    'HLT_Dimuon24_Upsilon_noCorrL1_v19',
    'HLT_Dimuon25_Jpsi_noCorrL1_v19',
    'HLT_Dimuon25_Jpsi_v27',
    'HLT_DoubleMu2_Jpsi_DoubleTrk1_Phi1p05_v19',
    'HLT_DoubleMu2_Jpsi_LowPt_v7',
    'HLT_DoubleMu3_DoubleEle7p5_CaloIdL_TrackIdL_Upsilon_v17',
    'HLT_DoubleMu3_TkMu_DsTau3Mu_v17',
    'HLT_DoubleMu3_Trk_Tau3mu_NoL1Mass_v19',
    'HLT_DoubleMu3_Trk_Tau3mu_v25',
    'HLT_DoubleMu4_3_Bs_v28',
    'HLT_DoubleMu4_3_Displaced_Photon4_BsToMMG_v13',
    'HLT_DoubleMu4_3_Jpsi_v28',
    'HLT_DoubleMu4_3_LowMass_SS_v7',
    'HLT_DoubleMu4_3_LowMass_v14',
    'HLT_DoubleMu4_3_Photon4_BsToMMG_v13',
    'HLT_DoubleMu4_JpsiTrkTrk_Displaced_v20',
    'HLT_DoubleMu4_JpsiTrk_Bc_v13',
    'HLT_DoubleMu4_Jpsi_Displaced_v20',
    'HLT_DoubleMu4_Jpsi_NoVertexing_v20',
    'HLT_DoubleMu4_LowMass_Displaced_v14',
    'HLT_DoubleMu4_MuMuTrk_Displaced_v28',
    'HLT_DoubleMu5_Upsilon_DoubleEle3_CaloIdL_TrackIdL_v17',
    'HLT_Mu25_TkMu0_Phi_v21',
    'HLT_Mu30_TkMu0_Psi_v14',
    'HLT_Mu30_TkMu0_Upsilon_v14',
    'HLT_Mu4_L1DoubleMu_v14',
    'HLT_Mu7p5_L2Mu2_Jpsi_v23',
    'HLT_Mu7p5_L2Mu2_Upsilon_v23',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_Charge1_v17',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_v17',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15_Charge1_v17',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15_v17',
    'HLT_Trimuon5_3p5_2_Upsilon_Muon_v18',
    'HLT_TrimuonOpen_5_3p5_2_Upsilon_Muon_v15'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingDoubleMuonLowMass1_datasetParkingDoubleMuonLowMass3_selector
streamParkingDoubleMuonLowMass1_datasetParkingDoubleMuonLowMass3_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingDoubleMuonLowMass1_datasetParkingDoubleMuonLowMass3_selector.l1tResults = cms.InputTag('')
streamParkingDoubleMuonLowMass1_datasetParkingDoubleMuonLowMass3_selector.throw      = cms.bool(False)
streamParkingDoubleMuonLowMass1_datasetParkingDoubleMuonLowMass3_selector.triggerConditions = cms.vstring(
    'HLT_Dimuon0_Jpsi3p5_Muon2_v18',
    'HLT_Dimuon0_Jpsi_L1_4R_0er1p5R_v20',
    'HLT_Dimuon0_Jpsi_L1_NoOS_v20',
    'HLT_Dimuon0_Jpsi_NoVertexing_L1_4R_0er1p5R_v20',
    'HLT_Dimuon0_Jpsi_NoVertexing_NoOS_v20',
    'HLT_Dimuon0_Jpsi_NoVertexing_v21',
    'HLT_Dimuon0_Jpsi_v21',
    'HLT_Dimuon0_LowMass_L1_0er1p5_v21',
    'HLT_Dimuon0_LowMass_L1_4_v21',
    'HLT_Dimuon0_LowMass_L1_TM530_v19',
    'HLT_Dimuon0_LowMass_v21',
    'HLT_Dimuon0_Upsilon_L1_4p5_v22',
    'HLT_Dimuon0_Upsilon_L1_4p5er2p0M_v20',
    'HLT_Dimuon0_Upsilon_L1_4p5er2p0_v22',
    'HLT_Dimuon0_Upsilon_Muon_NoL1Mass_v19',
    'HLT_Dimuon0_Upsilon_NoVertexing_v20',
    'HLT_Dimuon10_Upsilon_y1p4_v14',
    'HLT_Dimuon12_Upsilon_y1p4_v15',
    'HLT_Dimuon14_Phi_Barrel_Seagulls_v20',
    'HLT_Dimuon14_PsiPrime_noCorrL1_v18',
    'HLT_Dimuon14_PsiPrime_v26',
    'HLT_Dimuon18_PsiPrime_noCorrL1_v19',
    'HLT_Dimuon18_PsiPrime_v27',
    'HLT_Dimuon24_Phi_noCorrL1_v19',
    'HLT_Dimuon24_Upsilon_noCorrL1_v19',
    'HLT_Dimuon25_Jpsi_noCorrL1_v19',
    'HLT_Dimuon25_Jpsi_v27',
    'HLT_DoubleMu2_Jpsi_DoubleTrk1_Phi1p05_v19',
    'HLT_DoubleMu2_Jpsi_LowPt_v7',
    'HLT_DoubleMu3_DoubleEle7p5_CaloIdL_TrackIdL_Upsilon_v17',
    'HLT_DoubleMu3_TkMu_DsTau3Mu_v17',
    'HLT_DoubleMu3_Trk_Tau3mu_NoL1Mass_v19',
    'HLT_DoubleMu3_Trk_Tau3mu_v25',
    'HLT_DoubleMu4_3_Bs_v28',
    'HLT_DoubleMu4_3_Displaced_Photon4_BsToMMG_v13',
    'HLT_DoubleMu4_3_Jpsi_v28',
    'HLT_DoubleMu4_3_LowMass_SS_v7',
    'HLT_DoubleMu4_3_LowMass_v14',
    'HLT_DoubleMu4_3_Photon4_BsToMMG_v13',
    'HLT_DoubleMu4_JpsiTrkTrk_Displaced_v20',
    'HLT_DoubleMu4_JpsiTrk_Bc_v13',
    'HLT_DoubleMu4_Jpsi_Displaced_v20',
    'HLT_DoubleMu4_Jpsi_NoVertexing_v20',
    'HLT_DoubleMu4_LowMass_Displaced_v14',
    'HLT_DoubleMu4_MuMuTrk_Displaced_v28',
    'HLT_DoubleMu5_Upsilon_DoubleEle3_CaloIdL_TrackIdL_v17',
    'HLT_Mu25_TkMu0_Phi_v21',
    'HLT_Mu30_TkMu0_Psi_v14',
    'HLT_Mu30_TkMu0_Upsilon_v14',
    'HLT_Mu4_L1DoubleMu_v14',
    'HLT_Mu7p5_L2Mu2_Jpsi_v23',
    'HLT_Mu7p5_L2Mu2_Upsilon_v23',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_Charge1_v17',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_v17',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15_Charge1_v17',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15_v17',
    'HLT_Trimuon5_3p5_2_Upsilon_Muon_v18',
    'HLT_TrimuonOpen_5_3p5_2_Upsilon_Muon_v15'
)


# stream ParkingDoubleMuonLowMass2

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingDoubleMuonLowMass2_datasetParkingDoubleMuonLowMass4_selector
streamParkingDoubleMuonLowMass2_datasetParkingDoubleMuonLowMass4_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingDoubleMuonLowMass2_datasetParkingDoubleMuonLowMass4_selector.l1tResults = cms.InputTag('')
streamParkingDoubleMuonLowMass2_datasetParkingDoubleMuonLowMass4_selector.throw      = cms.bool(False)
streamParkingDoubleMuonLowMass2_datasetParkingDoubleMuonLowMass4_selector.triggerConditions = cms.vstring(
    'HLT_Dimuon0_Jpsi3p5_Muon2_v18',
    'HLT_Dimuon0_Jpsi_L1_4R_0er1p5R_v20',
    'HLT_Dimuon0_Jpsi_L1_NoOS_v20',
    'HLT_Dimuon0_Jpsi_NoVertexing_L1_4R_0er1p5R_v20',
    'HLT_Dimuon0_Jpsi_NoVertexing_NoOS_v20',
    'HLT_Dimuon0_Jpsi_NoVertexing_v21',
    'HLT_Dimuon0_Jpsi_v21',
    'HLT_Dimuon0_LowMass_L1_0er1p5_v21',
    'HLT_Dimuon0_LowMass_L1_4_v21',
    'HLT_Dimuon0_LowMass_L1_TM530_v19',
    'HLT_Dimuon0_LowMass_v21',
    'HLT_Dimuon0_Upsilon_L1_4p5_v22',
    'HLT_Dimuon0_Upsilon_L1_4p5er2p0M_v20',
    'HLT_Dimuon0_Upsilon_L1_4p5er2p0_v22',
    'HLT_Dimuon0_Upsilon_Muon_NoL1Mass_v19',
    'HLT_Dimuon0_Upsilon_NoVertexing_v20',
    'HLT_Dimuon10_Upsilon_y1p4_v14',
    'HLT_Dimuon12_Upsilon_y1p4_v15',
    'HLT_Dimuon14_Phi_Barrel_Seagulls_v20',
    'HLT_Dimuon14_PsiPrime_noCorrL1_v18',
    'HLT_Dimuon14_PsiPrime_v26',
    'HLT_Dimuon18_PsiPrime_noCorrL1_v19',
    'HLT_Dimuon18_PsiPrime_v27',
    'HLT_Dimuon24_Phi_noCorrL1_v19',
    'HLT_Dimuon24_Upsilon_noCorrL1_v19',
    'HLT_Dimuon25_Jpsi_noCorrL1_v19',
    'HLT_Dimuon25_Jpsi_v27',
    'HLT_DoubleMu2_Jpsi_DoubleTrk1_Phi1p05_v19',
    'HLT_DoubleMu2_Jpsi_LowPt_v7',
    'HLT_DoubleMu3_DoubleEle7p5_CaloIdL_TrackIdL_Upsilon_v17',
    'HLT_DoubleMu3_TkMu_DsTau3Mu_v17',
    'HLT_DoubleMu3_Trk_Tau3mu_NoL1Mass_v19',
    'HLT_DoubleMu3_Trk_Tau3mu_v25',
    'HLT_DoubleMu4_3_Bs_v28',
    'HLT_DoubleMu4_3_Displaced_Photon4_BsToMMG_v13',
    'HLT_DoubleMu4_3_Jpsi_v28',
    'HLT_DoubleMu4_3_LowMass_SS_v7',
    'HLT_DoubleMu4_3_LowMass_v14',
    'HLT_DoubleMu4_3_Photon4_BsToMMG_v13',
    'HLT_DoubleMu4_JpsiTrkTrk_Displaced_v20',
    'HLT_DoubleMu4_JpsiTrk_Bc_v13',
    'HLT_DoubleMu4_Jpsi_Displaced_v20',
    'HLT_DoubleMu4_Jpsi_NoVertexing_v20',
    'HLT_DoubleMu4_LowMass_Displaced_v14',
    'HLT_DoubleMu4_MuMuTrk_Displaced_v28',
    'HLT_DoubleMu5_Upsilon_DoubleEle3_CaloIdL_TrackIdL_v17',
    'HLT_Mu25_TkMu0_Phi_v21',
    'HLT_Mu30_TkMu0_Psi_v14',
    'HLT_Mu30_TkMu0_Upsilon_v14',
    'HLT_Mu4_L1DoubleMu_v14',
    'HLT_Mu7p5_L2Mu2_Jpsi_v23',
    'HLT_Mu7p5_L2Mu2_Upsilon_v23',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_Charge1_v17',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_v17',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15_Charge1_v17',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15_v17',
    'HLT_Trimuon5_3p5_2_Upsilon_Muon_v18',
    'HLT_TrimuonOpen_5_3p5_2_Upsilon_Muon_v15'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingDoubleMuonLowMass2_datasetParkingDoubleMuonLowMass5_selector
streamParkingDoubleMuonLowMass2_datasetParkingDoubleMuonLowMass5_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingDoubleMuonLowMass2_datasetParkingDoubleMuonLowMass5_selector.l1tResults = cms.InputTag('')
streamParkingDoubleMuonLowMass2_datasetParkingDoubleMuonLowMass5_selector.throw      = cms.bool(False)
streamParkingDoubleMuonLowMass2_datasetParkingDoubleMuonLowMass5_selector.triggerConditions = cms.vstring(
    'HLT_Dimuon0_Jpsi3p5_Muon2_v18',
    'HLT_Dimuon0_Jpsi_L1_4R_0er1p5R_v20',
    'HLT_Dimuon0_Jpsi_L1_NoOS_v20',
    'HLT_Dimuon0_Jpsi_NoVertexing_L1_4R_0er1p5R_v20',
    'HLT_Dimuon0_Jpsi_NoVertexing_NoOS_v20',
    'HLT_Dimuon0_Jpsi_NoVertexing_v21',
    'HLT_Dimuon0_Jpsi_v21',
    'HLT_Dimuon0_LowMass_L1_0er1p5_v21',
    'HLT_Dimuon0_LowMass_L1_4_v21',
    'HLT_Dimuon0_LowMass_L1_TM530_v19',
    'HLT_Dimuon0_LowMass_v21',
    'HLT_Dimuon0_Upsilon_L1_4p5_v22',
    'HLT_Dimuon0_Upsilon_L1_4p5er2p0M_v20',
    'HLT_Dimuon0_Upsilon_L1_4p5er2p0_v22',
    'HLT_Dimuon0_Upsilon_Muon_NoL1Mass_v19',
    'HLT_Dimuon0_Upsilon_NoVertexing_v20',
    'HLT_Dimuon10_Upsilon_y1p4_v14',
    'HLT_Dimuon12_Upsilon_y1p4_v15',
    'HLT_Dimuon14_Phi_Barrel_Seagulls_v20',
    'HLT_Dimuon14_PsiPrime_noCorrL1_v18',
    'HLT_Dimuon14_PsiPrime_v26',
    'HLT_Dimuon18_PsiPrime_noCorrL1_v19',
    'HLT_Dimuon18_PsiPrime_v27',
    'HLT_Dimuon24_Phi_noCorrL1_v19',
    'HLT_Dimuon24_Upsilon_noCorrL1_v19',
    'HLT_Dimuon25_Jpsi_noCorrL1_v19',
    'HLT_Dimuon25_Jpsi_v27',
    'HLT_DoubleMu2_Jpsi_DoubleTrk1_Phi1p05_v19',
    'HLT_DoubleMu2_Jpsi_LowPt_v7',
    'HLT_DoubleMu3_DoubleEle7p5_CaloIdL_TrackIdL_Upsilon_v17',
    'HLT_DoubleMu3_TkMu_DsTau3Mu_v17',
    'HLT_DoubleMu3_Trk_Tau3mu_NoL1Mass_v19',
    'HLT_DoubleMu3_Trk_Tau3mu_v25',
    'HLT_DoubleMu4_3_Bs_v28',
    'HLT_DoubleMu4_3_Displaced_Photon4_BsToMMG_v13',
    'HLT_DoubleMu4_3_Jpsi_v28',
    'HLT_DoubleMu4_3_LowMass_SS_v7',
    'HLT_DoubleMu4_3_LowMass_v14',
    'HLT_DoubleMu4_3_Photon4_BsToMMG_v13',
    'HLT_DoubleMu4_JpsiTrkTrk_Displaced_v20',
    'HLT_DoubleMu4_JpsiTrk_Bc_v13',
    'HLT_DoubleMu4_Jpsi_Displaced_v20',
    'HLT_DoubleMu4_Jpsi_NoVertexing_v20',
    'HLT_DoubleMu4_LowMass_Displaced_v14',
    'HLT_DoubleMu4_MuMuTrk_Displaced_v28',
    'HLT_DoubleMu5_Upsilon_DoubleEle3_CaloIdL_TrackIdL_v17',
    'HLT_Mu25_TkMu0_Phi_v21',
    'HLT_Mu30_TkMu0_Psi_v14',
    'HLT_Mu30_TkMu0_Upsilon_v14',
    'HLT_Mu4_L1DoubleMu_v14',
    'HLT_Mu7p5_L2Mu2_Jpsi_v23',
    'HLT_Mu7p5_L2Mu2_Upsilon_v23',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_Charge1_v17',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_v17',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15_Charge1_v17',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15_v17',
    'HLT_Trimuon5_3p5_2_Upsilon_Muon_v18',
    'HLT_TrimuonOpen_5_3p5_2_Upsilon_Muon_v15'
)


# stream ParkingDoubleMuonLowMass3

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingDoubleMuonLowMass3_datasetParkingDoubleMuonLowMass6_selector
streamParkingDoubleMuonLowMass3_datasetParkingDoubleMuonLowMass6_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingDoubleMuonLowMass3_datasetParkingDoubleMuonLowMass6_selector.l1tResults = cms.InputTag('')
streamParkingDoubleMuonLowMass3_datasetParkingDoubleMuonLowMass6_selector.throw      = cms.bool(False)
streamParkingDoubleMuonLowMass3_datasetParkingDoubleMuonLowMass6_selector.triggerConditions = cms.vstring(
    'HLT_Dimuon0_Jpsi3p5_Muon2_v18',
    'HLT_Dimuon0_Jpsi_L1_4R_0er1p5R_v20',
    'HLT_Dimuon0_Jpsi_L1_NoOS_v20',
    'HLT_Dimuon0_Jpsi_NoVertexing_L1_4R_0er1p5R_v20',
    'HLT_Dimuon0_Jpsi_NoVertexing_NoOS_v20',
    'HLT_Dimuon0_Jpsi_NoVertexing_v21',
    'HLT_Dimuon0_Jpsi_v21',
    'HLT_Dimuon0_LowMass_L1_0er1p5_v21',
    'HLT_Dimuon0_LowMass_L1_4_v21',
    'HLT_Dimuon0_LowMass_L1_TM530_v19',
    'HLT_Dimuon0_LowMass_v21',
    'HLT_Dimuon0_Upsilon_L1_4p5_v22',
    'HLT_Dimuon0_Upsilon_L1_4p5er2p0M_v20',
    'HLT_Dimuon0_Upsilon_L1_4p5er2p0_v22',
    'HLT_Dimuon0_Upsilon_Muon_NoL1Mass_v19',
    'HLT_Dimuon0_Upsilon_NoVertexing_v20',
    'HLT_Dimuon10_Upsilon_y1p4_v14',
    'HLT_Dimuon12_Upsilon_y1p4_v15',
    'HLT_Dimuon14_Phi_Barrel_Seagulls_v20',
    'HLT_Dimuon14_PsiPrime_noCorrL1_v18',
    'HLT_Dimuon14_PsiPrime_v26',
    'HLT_Dimuon18_PsiPrime_noCorrL1_v19',
    'HLT_Dimuon18_PsiPrime_v27',
    'HLT_Dimuon24_Phi_noCorrL1_v19',
    'HLT_Dimuon24_Upsilon_noCorrL1_v19',
    'HLT_Dimuon25_Jpsi_noCorrL1_v19',
    'HLT_Dimuon25_Jpsi_v27',
    'HLT_DoubleMu2_Jpsi_DoubleTrk1_Phi1p05_v19',
    'HLT_DoubleMu2_Jpsi_LowPt_v7',
    'HLT_DoubleMu3_DoubleEle7p5_CaloIdL_TrackIdL_Upsilon_v17',
    'HLT_DoubleMu3_TkMu_DsTau3Mu_v17',
    'HLT_DoubleMu3_Trk_Tau3mu_NoL1Mass_v19',
    'HLT_DoubleMu3_Trk_Tau3mu_v25',
    'HLT_DoubleMu4_3_Bs_v28',
    'HLT_DoubleMu4_3_Displaced_Photon4_BsToMMG_v13',
    'HLT_DoubleMu4_3_Jpsi_v28',
    'HLT_DoubleMu4_3_LowMass_SS_v7',
    'HLT_DoubleMu4_3_LowMass_v14',
    'HLT_DoubleMu4_3_Photon4_BsToMMG_v13',
    'HLT_DoubleMu4_JpsiTrkTrk_Displaced_v20',
    'HLT_DoubleMu4_JpsiTrk_Bc_v13',
    'HLT_DoubleMu4_Jpsi_Displaced_v20',
    'HLT_DoubleMu4_Jpsi_NoVertexing_v20',
    'HLT_DoubleMu4_LowMass_Displaced_v14',
    'HLT_DoubleMu4_MuMuTrk_Displaced_v28',
    'HLT_DoubleMu5_Upsilon_DoubleEle3_CaloIdL_TrackIdL_v17',
    'HLT_Mu25_TkMu0_Phi_v21',
    'HLT_Mu30_TkMu0_Psi_v14',
    'HLT_Mu30_TkMu0_Upsilon_v14',
    'HLT_Mu4_L1DoubleMu_v14',
    'HLT_Mu7p5_L2Mu2_Jpsi_v23',
    'HLT_Mu7p5_L2Mu2_Upsilon_v23',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_Charge1_v17',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_v17',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15_Charge1_v17',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15_v17',
    'HLT_Trimuon5_3p5_2_Upsilon_Muon_v18',
    'HLT_TrimuonOpen_5_3p5_2_Upsilon_Muon_v15'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingDoubleMuonLowMass3_datasetParkingDoubleMuonLowMass7_selector
streamParkingDoubleMuonLowMass3_datasetParkingDoubleMuonLowMass7_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingDoubleMuonLowMass3_datasetParkingDoubleMuonLowMass7_selector.l1tResults = cms.InputTag('')
streamParkingDoubleMuonLowMass3_datasetParkingDoubleMuonLowMass7_selector.throw      = cms.bool(False)
streamParkingDoubleMuonLowMass3_datasetParkingDoubleMuonLowMass7_selector.triggerConditions = cms.vstring(
    'HLT_Dimuon0_Jpsi3p5_Muon2_v18',
    'HLT_Dimuon0_Jpsi_L1_4R_0er1p5R_v20',
    'HLT_Dimuon0_Jpsi_L1_NoOS_v20',
    'HLT_Dimuon0_Jpsi_NoVertexing_L1_4R_0er1p5R_v20',
    'HLT_Dimuon0_Jpsi_NoVertexing_NoOS_v20',
    'HLT_Dimuon0_Jpsi_NoVertexing_v21',
    'HLT_Dimuon0_Jpsi_v21',
    'HLT_Dimuon0_LowMass_L1_0er1p5_v21',
    'HLT_Dimuon0_LowMass_L1_4_v21',
    'HLT_Dimuon0_LowMass_L1_TM530_v19',
    'HLT_Dimuon0_LowMass_v21',
    'HLT_Dimuon0_Upsilon_L1_4p5_v22',
    'HLT_Dimuon0_Upsilon_L1_4p5er2p0M_v20',
    'HLT_Dimuon0_Upsilon_L1_4p5er2p0_v22',
    'HLT_Dimuon0_Upsilon_Muon_NoL1Mass_v19',
    'HLT_Dimuon0_Upsilon_NoVertexing_v20',
    'HLT_Dimuon10_Upsilon_y1p4_v14',
    'HLT_Dimuon12_Upsilon_y1p4_v15',
    'HLT_Dimuon14_Phi_Barrel_Seagulls_v20',
    'HLT_Dimuon14_PsiPrime_noCorrL1_v18',
    'HLT_Dimuon14_PsiPrime_v26',
    'HLT_Dimuon18_PsiPrime_noCorrL1_v19',
    'HLT_Dimuon18_PsiPrime_v27',
    'HLT_Dimuon24_Phi_noCorrL1_v19',
    'HLT_Dimuon24_Upsilon_noCorrL1_v19',
    'HLT_Dimuon25_Jpsi_noCorrL1_v19',
    'HLT_Dimuon25_Jpsi_v27',
    'HLT_DoubleMu2_Jpsi_DoubleTrk1_Phi1p05_v19',
    'HLT_DoubleMu2_Jpsi_LowPt_v7',
    'HLT_DoubleMu3_DoubleEle7p5_CaloIdL_TrackIdL_Upsilon_v17',
    'HLT_DoubleMu3_TkMu_DsTau3Mu_v17',
    'HLT_DoubleMu3_Trk_Tau3mu_NoL1Mass_v19',
    'HLT_DoubleMu3_Trk_Tau3mu_v25',
    'HLT_DoubleMu4_3_Bs_v28',
    'HLT_DoubleMu4_3_Displaced_Photon4_BsToMMG_v13',
    'HLT_DoubleMu4_3_Jpsi_v28',
    'HLT_DoubleMu4_3_LowMass_SS_v7',
    'HLT_DoubleMu4_3_LowMass_v14',
    'HLT_DoubleMu4_3_Photon4_BsToMMG_v13',
    'HLT_DoubleMu4_JpsiTrkTrk_Displaced_v20',
    'HLT_DoubleMu4_JpsiTrk_Bc_v13',
    'HLT_DoubleMu4_Jpsi_Displaced_v20',
    'HLT_DoubleMu4_Jpsi_NoVertexing_v20',
    'HLT_DoubleMu4_LowMass_Displaced_v14',
    'HLT_DoubleMu4_MuMuTrk_Displaced_v28',
    'HLT_DoubleMu5_Upsilon_DoubleEle3_CaloIdL_TrackIdL_v17',
    'HLT_Mu25_TkMu0_Phi_v21',
    'HLT_Mu30_TkMu0_Psi_v14',
    'HLT_Mu30_TkMu0_Upsilon_v14',
    'HLT_Mu4_L1DoubleMu_v14',
    'HLT_Mu7p5_L2Mu2_Jpsi_v23',
    'HLT_Mu7p5_L2Mu2_Upsilon_v23',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_Charge1_v17',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_v17',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15_Charge1_v17',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15_v17',
    'HLT_Trimuon5_3p5_2_Upsilon_Muon_v18',
    'HLT_TrimuonOpen_5_3p5_2_Upsilon_Muon_v15'
)


# stream ParkingHH

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingHH_datasetParkingHH0_selector
streamParkingHH_datasetParkingHH0_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingHH_datasetParkingHH0_selector.l1tResults = cms.InputTag('')
streamParkingHH_datasetParkingHH0_selector.throw      = cms.bool(False)
streamParkingHH_datasetParkingHH0_selector.triggerConditions = cms.vstring(
    'HLT_L1HT200_QuadPFJet20_PNet1BTag0p50_PNet1Tauh0p50_v1',
    'HLT_L1HT200_QuadPFJet25_PNet1BTag0p50_PNet1Tauh0p50_v1',
    'HLT_PFHT250_QuadPFJet25_PNet1BTag0p20_PNet1Tauh0p50_v7',
    'HLT_PFHT250_QuadPFJet25_PNet2BTagMean0p55_v7',
    'HLT_PFHT250_QuadPFJet25_v7',
    'HLT_PFHT250_QuadPFJet30_PNet1BTag0p20_PNet1Tauh0p50_v7',
    'HLT_PFHT250_QuadPFJet30_PNet2BTagMean0p55_v7',
    'HLT_PFHT280_QuadPFJet30_PNet1BTag0p20_PNet1Tauh0p50_v7',
    'HLT_PFHT280_QuadPFJet30_PNet2BTagMean0p55_v10',
    'HLT_PFHT280_QuadPFJet30_PNet2BTagMean0p60_v10',
    'HLT_PFHT280_QuadPFJet30_v10',
    'HLT_PFHT280_QuadPFJet35_PNet2BTagMean0p60_v10',
    'HLT_PFHT330PT30_QuadPFJet_75_60_45_40_PNet3BTag_2p0_v6',
    'HLT_PFHT330PT30_QuadPFJet_75_60_45_40_PNet3BTag_4p3_v6',
    'HLT_PFHT330PT30_QuadPFJet_75_60_45_40_v22',
    'HLT_PFHT340_QuadPFJet70_50_40_40_PNet2BTagMean0p70_v11',
    'HLT_PFHT400_FivePFJet_120_120_60_30_30_PNet2BTag_4p3_v7',
    'HLT_PFHT400_FivePFJet_120_120_60_30_30_PNet2BTag_5p6_v7',
    'HLT_PFHT400_FivePFJet_120_120_60_30_30_v7',
    'HLT_PFHT400_SixPFJet32_PNet2BTagMean0p50_v10',
    'HLT_PFHT400_SixPFJet32_v22',
    'HLT_PFHT450_SixPFJet36_PNetBTag0p35_v10',
    'HLT_PFHT450_SixPFJet36_v21'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingHH_datasetParkingHH1_selector
streamParkingHH_datasetParkingHH1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingHH_datasetParkingHH1_selector.l1tResults = cms.InputTag('')
streamParkingHH_datasetParkingHH1_selector.throw      = cms.bool(False)
streamParkingHH_datasetParkingHH1_selector.triggerConditions = cms.vstring(
    'HLT_L1HT200_QuadPFJet20_PNet1BTag0p50_PNet1Tauh0p50_v1',
    'HLT_L1HT200_QuadPFJet25_PNet1BTag0p50_PNet1Tauh0p50_v1',
    'HLT_PFHT250_QuadPFJet25_PNet1BTag0p20_PNet1Tauh0p50_v7',
    'HLT_PFHT250_QuadPFJet25_PNet2BTagMean0p55_v7',
    'HLT_PFHT250_QuadPFJet25_v7',
    'HLT_PFHT250_QuadPFJet30_PNet1BTag0p20_PNet1Tauh0p50_v7',
    'HLT_PFHT250_QuadPFJet30_PNet2BTagMean0p55_v7',
    'HLT_PFHT280_QuadPFJet30_PNet1BTag0p20_PNet1Tauh0p50_v7',
    'HLT_PFHT280_QuadPFJet30_PNet2BTagMean0p55_v10',
    'HLT_PFHT280_QuadPFJet30_PNet2BTagMean0p60_v10',
    'HLT_PFHT280_QuadPFJet30_v10',
    'HLT_PFHT280_QuadPFJet35_PNet2BTagMean0p60_v10',
    'HLT_PFHT330PT30_QuadPFJet_75_60_45_40_PNet3BTag_2p0_v6',
    'HLT_PFHT330PT30_QuadPFJet_75_60_45_40_PNet3BTag_4p3_v6',
    'HLT_PFHT330PT30_QuadPFJet_75_60_45_40_v22',
    'HLT_PFHT340_QuadPFJet70_50_40_40_PNet2BTagMean0p70_v11',
    'HLT_PFHT400_FivePFJet_120_120_60_30_30_PNet2BTag_4p3_v7',
    'HLT_PFHT400_FivePFJet_120_120_60_30_30_PNet2BTag_5p6_v7',
    'HLT_PFHT400_FivePFJet_120_120_60_30_30_v7',
    'HLT_PFHT400_SixPFJet32_PNet2BTagMean0p50_v10',
    'HLT_PFHT400_SixPFJet32_v22',
    'HLT_PFHT450_SixPFJet36_PNetBTag0p35_v10',
    'HLT_PFHT450_SixPFJet36_v21'
)


# stream ParkingLLP

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingLLP_datasetParkingLLP0_selector
streamParkingLLP_datasetParkingLLP0_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingLLP_datasetParkingLLP0_selector.l1tResults = cms.InputTag('')
streamParkingLLP_datasetParkingLLP0_selector.throw      = cms.bool(False)
streamParkingLLP_datasetParkingLLP0_selector.triggerConditions = cms.vstring(
    'HLT_CaloMET60_DTCluster50_v12',
    'HLT_CaloMET60_DTClusterNoMB1S50_v12',
    'HLT_CscCluster150_DisplacedSingleJet30_Inclusive1PtrkShortSig5_v1',
    'HLT_CscCluster150_DisplacedSingleJet35_Inclusive1PtrkShortSig5_v1',
    'HLT_CscCluster150_DisplacedSingleJet40_Inclusive1PtrkShortSig5_v1',
    'HLT_HT170_L1SingleLLPJet_DisplacedDijet40_DisplacedTrack_v12',
    'HLT_HT200_L1SingleLLPJet_DelayedJet40_DoubleDelay0p5nsTrackless_v12',
    'HLT_HT200_L1SingleLLPJet_DelayedJet40_DoubleDelay1nsInclusive_v12',
    'HLT_HT200_L1SingleLLPJet_DelayedJet40_SingleDelay1nsTrackless_v12',
    'HLT_HT200_L1SingleLLPJet_DelayedJet40_SingleDelay2nsInclusive_v12',
    'HLT_HT200_L1SingleLLPJet_DisplacedDijet40_DisplacedTrack_v12',
    'HLT_HT200_L1SingleLLPJet_DisplacedDijet40_Inclusive1PtrkShortSig5_v12',
    'HLT_HT200_L1SingleLLPJet_DisplacedDijet60_DisplacedTrack_v12',
    'HLT_HT200_L1SingleLLPJet_PFJet60_NeutralHadronFrac0p7_v7',
    'HLT_HT200_L1SingleLLPJet_PFJet60_NeutralHadronFrac0p8_v7',
    'HLT_HT240_L1SingleLLPJet_DisplacedDijet40_Inclusive1PtrkShortSig5_v9',
    'HLT_HT270_L1SingleLLPJet_DisplacedDijet40_DisplacedTrack_v12',
    'HLT_HT280_L1SingleLLPJet_DisplacedDijet40_Inclusive1PtrkShortSig5_v9',
    'HLT_HT320_L1SingleLLPJet_DisplacedDijet60_Inclusive_v12',
    'HLT_HT350_DelayedJet40_SingleDelay1p5To3p5nsInclusive_v8',
    'HLT_HT350_DelayedJet40_SingleDelay1p6To3p5nsInclusive_v8',
    'HLT_HT350_DelayedJet40_SingleDelay1p75To3p5nsInclusive_v8',
    'HLT_HT350_DelayedJet40_SingleDelay3nsInclusive_v8',
    'HLT_HT350_DelayedJet40_SingleDelay3p25nsInclusive_v8',
    'HLT_HT350_DelayedJet40_SingleDelay3p5nsInclusive_v8',
    'HLT_HT360_DisplacedDijet40_Inclusive1PtrkShortSig5_v8',
    'HLT_HT360_DisplacedDijet45_Inclusive1PtrkShortSig5_v8',
    'HLT_HT390_DisplacedDijet40_Inclusive1PtrkShortSig5_v8',
    'HLT_HT390_DisplacedDijet45_Inclusive1PtrkShortSig5_v8',
    'HLT_HT390eta2p0_DisplacedDijet40_Inclusive1PtrkShortSig5_v8',
    'HLT_HT400_DisplacedDijet40_DisplacedTrack_v24',
    'HLT_HT420_L1SingleLLPJet_DisplacedDijet60_Inclusive_v12',
    'HLT_HT430_DelayedJet40_DoubleDelay0p5nsInclusive_v11',
    'HLT_HT430_DelayedJet40_DoubleDelay0p5nsTrackless_v12',
    'HLT_HT430_DelayedJet40_DoubleDelay0p75nsTrackless_v8',
    'HLT_HT430_DelayedJet40_DoubleDelay1nsInclusive_v12',
    'HLT_HT430_DelayedJet40_DoubleDelay1nsTrackless_v8',
    'HLT_HT430_DelayedJet40_DoubleDelay1p25nsInclusive_v8',
    'HLT_HT430_DelayedJet40_DoubleDelay1p5nsInclusive_v8',
    'HLT_HT430_DelayedJet40_SingleDelay0p5nsInclusive_v10',
    'HLT_HT430_DelayedJet40_SingleDelay0p5nsTrackless_v10',
    'HLT_HT430_DelayedJet40_SingleDelay1To1p5nsInclusive_v8',
    'HLT_HT430_DelayedJet40_SingleDelay1nsInclusive_v10',
    'HLT_HT430_DelayedJet40_SingleDelay1nsTrackless_v12',
    'HLT_HT430_DelayedJet40_SingleDelay1p1To1p6nsInclusive_v8',
    'HLT_HT430_DelayedJet40_SingleDelay1p25To1p75nsInclusive_v8',
    'HLT_HT430_DelayedJet40_SingleDelay1p25nsTrackless_v8',
    'HLT_HT430_DelayedJet40_SingleDelay1p5nsInclusive_v10',
    'HLT_HT430_DelayedJet40_SingleDelay1p5nsTrackless_v8',
    'HLT_HT430_DelayedJet40_SingleDelay2nsInclusive_v12',
    'HLT_HT430_DelayedJet40_SingleDelay2p25nsInclusive_v8',
    'HLT_HT430_DelayedJet40_SingleDelay2p5nsInclusive_v8',
    'HLT_HT430_DisplacedDijet40_DisplacedTrack_v24',
    'HLT_HT430_DisplacedDijet40_Inclusive1PtrkShortSig5_v12',
    'HLT_HT550_DisplacedDijet60_Inclusive_v24',
    'HLT_HT650_DisplacedDijet60_Inclusive_v24',
    'HLT_L1MET_DTCluster50_v12',
    'HLT_L1MET_DTClusterNoMB1S50_v12',
    'HLT_L1SingleLLPJet_v8',
    'HLT_L1Tau_DelayedJet40_DoubleDelay0p5nsTrackless_v10',
    'HLT_L1Tau_DelayedJet40_DoubleDelay0p75nsInclusive_v10',
    'HLT_L1Tau_DelayedJet40_DoubleDelay1nsTrackless_v10',
    'HLT_L1Tau_DelayedJet40_DoubleDelay1p25nsInclusive_v10',
    'HLT_L1Tau_DelayedJet40_DoubleDelay1p25nsTrackless_v8',
    'HLT_L1Tau_DelayedJet40_DoubleDelay1p5nsInclusive_v8',
    'HLT_L1Tau_DelayedJet40_DoubleDelay1p5nsTrackless_v8',
    'HLT_L1Tau_DelayedJet40_DoubleDelay1p75nsInclusive_v8',
    'HLT_L1Tau_DelayedJet40_SingleDelay2p5To4nsInclusive_v8',
    'HLT_L1Tau_DelayedJet40_SingleDelay2p5nsTrackless_v10',
    'HLT_L1Tau_DelayedJet40_SingleDelay2p6To4nsInclusive_v8',
    'HLT_L1Tau_DelayedJet40_SingleDelay2p75To4nsInclusive_v8',
    'HLT_L1Tau_DelayedJet40_SingleDelay2p75nsTrackless_v8',
    'HLT_L1Tau_DelayedJet40_SingleDelay3nsTrackless_v8',
    'HLT_L1Tau_DelayedJet40_SingleDelay3p5nsInclusive_v10',
    'HLT_L1Tau_DelayedJet40_SingleDelay3p75nsInclusive_v8',
    'HLT_L1Tau_DelayedJet40_SingleDelay4nsInclusive_v8',
    'HLT_Mu6HT240_DisplacedDijet30_Inclusive1PtrkShortSig5_DisplacedLoose_v12',
    'HLT_Mu6HT240_DisplacedDijet35_Inclusive0PtrkShortSig5_v12',
    'HLT_Mu6HT240_DisplacedDijet35_Inclusive1PtrkShortSig5_DisplacedLoose_v12',
    'HLT_Mu6HT240_DisplacedDijet40_Inclusive0PtrkShortSig5_v12',
    'HLT_Mu6HT240_DisplacedDijet40_Inclusive1PtrkShortSig5_DisplacedLoose_v12',
    'HLT_Mu6HT240_DisplacedDijet45_Inclusive0PtrkShortSig5_v12',
    'HLT_Mu6HT240_DisplacedDijet50_Inclusive0PtrkShortSig5_v12',
    'HLT_PFJet200_TimeGt2p5ns_v11',
    'HLT_PFJet200_TimeLtNeg2p5ns_v11'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingLLP_datasetParkingLLP1_selector
streamParkingLLP_datasetParkingLLP1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingLLP_datasetParkingLLP1_selector.l1tResults = cms.InputTag('')
streamParkingLLP_datasetParkingLLP1_selector.throw      = cms.bool(False)
streamParkingLLP_datasetParkingLLP1_selector.triggerConditions = cms.vstring(
    'HLT_CaloMET60_DTCluster50_v12',
    'HLT_CaloMET60_DTClusterNoMB1S50_v12',
    'HLT_CscCluster150_DisplacedSingleJet30_Inclusive1PtrkShortSig5_v1',
    'HLT_CscCluster150_DisplacedSingleJet35_Inclusive1PtrkShortSig5_v1',
    'HLT_CscCluster150_DisplacedSingleJet40_Inclusive1PtrkShortSig5_v1',
    'HLT_HT170_L1SingleLLPJet_DisplacedDijet40_DisplacedTrack_v12',
    'HLT_HT200_L1SingleLLPJet_DelayedJet40_DoubleDelay0p5nsTrackless_v12',
    'HLT_HT200_L1SingleLLPJet_DelayedJet40_DoubleDelay1nsInclusive_v12',
    'HLT_HT200_L1SingleLLPJet_DelayedJet40_SingleDelay1nsTrackless_v12',
    'HLT_HT200_L1SingleLLPJet_DelayedJet40_SingleDelay2nsInclusive_v12',
    'HLT_HT200_L1SingleLLPJet_DisplacedDijet40_DisplacedTrack_v12',
    'HLT_HT200_L1SingleLLPJet_DisplacedDijet40_Inclusive1PtrkShortSig5_v12',
    'HLT_HT200_L1SingleLLPJet_DisplacedDijet60_DisplacedTrack_v12',
    'HLT_HT200_L1SingleLLPJet_PFJet60_NeutralHadronFrac0p7_v7',
    'HLT_HT200_L1SingleLLPJet_PFJet60_NeutralHadronFrac0p8_v7',
    'HLT_HT240_L1SingleLLPJet_DisplacedDijet40_Inclusive1PtrkShortSig5_v9',
    'HLT_HT270_L1SingleLLPJet_DisplacedDijet40_DisplacedTrack_v12',
    'HLT_HT280_L1SingleLLPJet_DisplacedDijet40_Inclusive1PtrkShortSig5_v9',
    'HLT_HT320_L1SingleLLPJet_DisplacedDijet60_Inclusive_v12',
    'HLT_HT350_DelayedJet40_SingleDelay1p5To3p5nsInclusive_v8',
    'HLT_HT350_DelayedJet40_SingleDelay1p6To3p5nsInclusive_v8',
    'HLT_HT350_DelayedJet40_SingleDelay1p75To3p5nsInclusive_v8',
    'HLT_HT350_DelayedJet40_SingleDelay3nsInclusive_v8',
    'HLT_HT350_DelayedJet40_SingleDelay3p25nsInclusive_v8',
    'HLT_HT350_DelayedJet40_SingleDelay3p5nsInclusive_v8',
    'HLT_HT360_DisplacedDijet40_Inclusive1PtrkShortSig5_v8',
    'HLT_HT360_DisplacedDijet45_Inclusive1PtrkShortSig5_v8',
    'HLT_HT390_DisplacedDijet40_Inclusive1PtrkShortSig5_v8',
    'HLT_HT390_DisplacedDijet45_Inclusive1PtrkShortSig5_v8',
    'HLT_HT390eta2p0_DisplacedDijet40_Inclusive1PtrkShortSig5_v8',
    'HLT_HT400_DisplacedDijet40_DisplacedTrack_v24',
    'HLT_HT420_L1SingleLLPJet_DisplacedDijet60_Inclusive_v12',
    'HLT_HT430_DelayedJet40_DoubleDelay0p5nsInclusive_v11',
    'HLT_HT430_DelayedJet40_DoubleDelay0p5nsTrackless_v12',
    'HLT_HT430_DelayedJet40_DoubleDelay0p75nsTrackless_v8',
    'HLT_HT430_DelayedJet40_DoubleDelay1nsInclusive_v12',
    'HLT_HT430_DelayedJet40_DoubleDelay1nsTrackless_v8',
    'HLT_HT430_DelayedJet40_DoubleDelay1p25nsInclusive_v8',
    'HLT_HT430_DelayedJet40_DoubleDelay1p5nsInclusive_v8',
    'HLT_HT430_DelayedJet40_SingleDelay0p5nsInclusive_v10',
    'HLT_HT430_DelayedJet40_SingleDelay0p5nsTrackless_v10',
    'HLT_HT430_DelayedJet40_SingleDelay1To1p5nsInclusive_v8',
    'HLT_HT430_DelayedJet40_SingleDelay1nsInclusive_v10',
    'HLT_HT430_DelayedJet40_SingleDelay1nsTrackless_v12',
    'HLT_HT430_DelayedJet40_SingleDelay1p1To1p6nsInclusive_v8',
    'HLT_HT430_DelayedJet40_SingleDelay1p25To1p75nsInclusive_v8',
    'HLT_HT430_DelayedJet40_SingleDelay1p25nsTrackless_v8',
    'HLT_HT430_DelayedJet40_SingleDelay1p5nsInclusive_v10',
    'HLT_HT430_DelayedJet40_SingleDelay1p5nsTrackless_v8',
    'HLT_HT430_DelayedJet40_SingleDelay2nsInclusive_v12',
    'HLT_HT430_DelayedJet40_SingleDelay2p25nsInclusive_v8',
    'HLT_HT430_DelayedJet40_SingleDelay2p5nsInclusive_v8',
    'HLT_HT430_DisplacedDijet40_DisplacedTrack_v24',
    'HLT_HT430_DisplacedDijet40_Inclusive1PtrkShortSig5_v12',
    'HLT_HT550_DisplacedDijet60_Inclusive_v24',
    'HLT_HT650_DisplacedDijet60_Inclusive_v24',
    'HLT_L1MET_DTCluster50_v12',
    'HLT_L1MET_DTClusterNoMB1S50_v12',
    'HLT_L1SingleLLPJet_v8',
    'HLT_L1Tau_DelayedJet40_DoubleDelay0p5nsTrackless_v10',
    'HLT_L1Tau_DelayedJet40_DoubleDelay0p75nsInclusive_v10',
    'HLT_L1Tau_DelayedJet40_DoubleDelay1nsTrackless_v10',
    'HLT_L1Tau_DelayedJet40_DoubleDelay1p25nsInclusive_v10',
    'HLT_L1Tau_DelayedJet40_DoubleDelay1p25nsTrackless_v8',
    'HLT_L1Tau_DelayedJet40_DoubleDelay1p5nsInclusive_v8',
    'HLT_L1Tau_DelayedJet40_DoubleDelay1p5nsTrackless_v8',
    'HLT_L1Tau_DelayedJet40_DoubleDelay1p75nsInclusive_v8',
    'HLT_L1Tau_DelayedJet40_SingleDelay2p5To4nsInclusive_v8',
    'HLT_L1Tau_DelayedJet40_SingleDelay2p5nsTrackless_v10',
    'HLT_L1Tau_DelayedJet40_SingleDelay2p6To4nsInclusive_v8',
    'HLT_L1Tau_DelayedJet40_SingleDelay2p75To4nsInclusive_v8',
    'HLT_L1Tau_DelayedJet40_SingleDelay2p75nsTrackless_v8',
    'HLT_L1Tau_DelayedJet40_SingleDelay3nsTrackless_v8',
    'HLT_L1Tau_DelayedJet40_SingleDelay3p5nsInclusive_v10',
    'HLT_L1Tau_DelayedJet40_SingleDelay3p75nsInclusive_v8',
    'HLT_L1Tau_DelayedJet40_SingleDelay4nsInclusive_v8',
    'HLT_Mu6HT240_DisplacedDijet30_Inclusive1PtrkShortSig5_DisplacedLoose_v12',
    'HLT_Mu6HT240_DisplacedDijet35_Inclusive0PtrkShortSig5_v12',
    'HLT_Mu6HT240_DisplacedDijet35_Inclusive1PtrkShortSig5_DisplacedLoose_v12',
    'HLT_Mu6HT240_DisplacedDijet40_Inclusive0PtrkShortSig5_v12',
    'HLT_Mu6HT240_DisplacedDijet40_Inclusive1PtrkShortSig5_DisplacedLoose_v12',
    'HLT_Mu6HT240_DisplacedDijet45_Inclusive0PtrkShortSig5_v12',
    'HLT_Mu6HT240_DisplacedDijet50_Inclusive0PtrkShortSig5_v12',
    'HLT_PFJet200_TimeGt2p5ns_v11',
    'HLT_PFJet200_TimeLtNeg2p5ns_v11'
)


# stream ParkingSingleMuon0

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingSingleMuon0_datasetParkingSingleMuon0_selector
streamParkingSingleMuon0_datasetParkingSingleMuon0_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingSingleMuon0_datasetParkingSingleMuon0_selector.l1tResults = cms.InputTag('')
streamParkingSingleMuon0_datasetParkingSingleMuon0_selector.throw      = cms.bool(False)
streamParkingSingleMuon0_datasetParkingSingleMuon0_selector.triggerConditions = cms.vstring(
    'HLT_Mu0_Barrel_L1HP10_v7',
    'HLT_Mu0_Barrel_L1HP11_v7',
    'HLT_Mu0_Barrel_L1HP13_v1',
    'HLT_Mu0_Barrel_L1HP6_IP6_v4',
    'HLT_Mu0_Barrel_L1HP6_v4',
    'HLT_Mu0_Barrel_L1HP7_v4',
    'HLT_Mu0_Barrel_L1HP8_v5',
    'HLT_Mu0_Barrel_L1HP9_v5',
    'HLT_Mu0_Barrel_v7',
    'HLT_Mu10_Barrel_L1HP11_IP4_v1',
    'HLT_Mu10_Barrel_L1HP11_IP6_v7',
    'HLT_Mu12_Barrel_L1HP13_IP4_v1',
    'HLT_Mu12_Barrel_L1HP13_IP6_v1',
    'HLT_Mu4_Barrel_IP4_v1',
    'HLT_Mu4_Barrel_IP6_v1',
    'HLT_Mu6_Barrel_L1HP7_IP6_v4',
    'HLT_Mu7_Barrel_L1HP8_IP6_v5',
    'HLT_Mu8_Barrel_L1HP9_IP6_v5',
    'HLT_Mu9_Barrel_L1HP10_IP6_v7'
)


# stream ParkingSingleMuon1

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingSingleMuon1_datasetParkingSingleMuon1_selector
streamParkingSingleMuon1_datasetParkingSingleMuon1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingSingleMuon1_datasetParkingSingleMuon1_selector.l1tResults = cms.InputTag('')
streamParkingSingleMuon1_datasetParkingSingleMuon1_selector.throw      = cms.bool(False)
streamParkingSingleMuon1_datasetParkingSingleMuon1_selector.triggerConditions = cms.vstring(
    'HLT_Mu0_Barrel_L1HP10_v7',
    'HLT_Mu0_Barrel_L1HP11_v7',
    'HLT_Mu0_Barrel_L1HP13_v1',
    'HLT_Mu0_Barrel_L1HP6_IP6_v4',
    'HLT_Mu0_Barrel_L1HP6_v4',
    'HLT_Mu0_Barrel_L1HP7_v4',
    'HLT_Mu0_Barrel_L1HP8_v5',
    'HLT_Mu0_Barrel_L1HP9_v5',
    'HLT_Mu0_Barrel_v7',
    'HLT_Mu10_Barrel_L1HP11_IP4_v1',
    'HLT_Mu10_Barrel_L1HP11_IP6_v7',
    'HLT_Mu12_Barrel_L1HP13_IP4_v1',
    'HLT_Mu12_Barrel_L1HP13_IP6_v1',
    'HLT_Mu4_Barrel_IP4_v1',
    'HLT_Mu4_Barrel_IP6_v1',
    'HLT_Mu6_Barrel_L1HP7_IP6_v4',
    'HLT_Mu7_Barrel_L1HP8_IP6_v5',
    'HLT_Mu8_Barrel_L1HP9_IP6_v5',
    'HLT_Mu9_Barrel_L1HP10_IP6_v7'
)


# stream ParkingSingleMuon10

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingSingleMuon10_datasetParkingSingleMuon10_selector
streamParkingSingleMuon10_datasetParkingSingleMuon10_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingSingleMuon10_datasetParkingSingleMuon10_selector.l1tResults = cms.InputTag('')
streamParkingSingleMuon10_datasetParkingSingleMuon10_selector.throw      = cms.bool(False)
streamParkingSingleMuon10_datasetParkingSingleMuon10_selector.triggerConditions = cms.vstring(
    'HLT_Mu0_Barrel_L1HP10_v7',
    'HLT_Mu0_Barrel_L1HP11_v7',
    'HLT_Mu0_Barrel_L1HP13_v1',
    'HLT_Mu0_Barrel_L1HP6_IP6_v4',
    'HLT_Mu0_Barrel_L1HP6_v4',
    'HLT_Mu0_Barrel_L1HP7_v4',
    'HLT_Mu0_Barrel_L1HP8_v5',
    'HLT_Mu0_Barrel_L1HP9_v5',
    'HLT_Mu0_Barrel_v7',
    'HLT_Mu10_Barrel_L1HP11_IP4_v1',
    'HLT_Mu10_Barrel_L1HP11_IP6_v7',
    'HLT_Mu12_Barrel_L1HP13_IP4_v1',
    'HLT_Mu12_Barrel_L1HP13_IP6_v1',
    'HLT_Mu4_Barrel_IP4_v1',
    'HLT_Mu4_Barrel_IP6_v1',
    'HLT_Mu6_Barrel_L1HP7_IP6_v4',
    'HLT_Mu7_Barrel_L1HP8_IP6_v5',
    'HLT_Mu8_Barrel_L1HP9_IP6_v5',
    'HLT_Mu9_Barrel_L1HP10_IP6_v7'
)


# stream ParkingSingleMuon11

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingSingleMuon11_datasetParkingSingleMuon11_selector
streamParkingSingleMuon11_datasetParkingSingleMuon11_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingSingleMuon11_datasetParkingSingleMuon11_selector.l1tResults = cms.InputTag('')
streamParkingSingleMuon11_datasetParkingSingleMuon11_selector.throw      = cms.bool(False)
streamParkingSingleMuon11_datasetParkingSingleMuon11_selector.triggerConditions = cms.vstring(
    'HLT_Mu0_Barrel_L1HP10_v7',
    'HLT_Mu0_Barrel_L1HP11_v7',
    'HLT_Mu0_Barrel_L1HP13_v1',
    'HLT_Mu0_Barrel_L1HP6_IP6_v4',
    'HLT_Mu0_Barrel_L1HP6_v4',
    'HLT_Mu0_Barrel_L1HP7_v4',
    'HLT_Mu0_Barrel_L1HP8_v5',
    'HLT_Mu0_Barrel_L1HP9_v5',
    'HLT_Mu0_Barrel_v7',
    'HLT_Mu10_Barrel_L1HP11_IP4_v1',
    'HLT_Mu10_Barrel_L1HP11_IP6_v7',
    'HLT_Mu12_Barrel_L1HP13_IP4_v1',
    'HLT_Mu12_Barrel_L1HP13_IP6_v1',
    'HLT_Mu4_Barrel_IP4_v1',
    'HLT_Mu4_Barrel_IP6_v1',
    'HLT_Mu6_Barrel_L1HP7_IP6_v4',
    'HLT_Mu7_Barrel_L1HP8_IP6_v5',
    'HLT_Mu8_Barrel_L1HP9_IP6_v5',
    'HLT_Mu9_Barrel_L1HP10_IP6_v7'
)


# stream ParkingSingleMuon12

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingSingleMuon12_datasetParkingSingleMuon12_selector
streamParkingSingleMuon12_datasetParkingSingleMuon12_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingSingleMuon12_datasetParkingSingleMuon12_selector.l1tResults = cms.InputTag('')
streamParkingSingleMuon12_datasetParkingSingleMuon12_selector.throw      = cms.bool(False)
streamParkingSingleMuon12_datasetParkingSingleMuon12_selector.triggerConditions = cms.vstring(
    'HLT_Mu0_Barrel_L1HP10_v7',
    'HLT_Mu0_Barrel_L1HP11_v7',
    'HLT_Mu0_Barrel_L1HP13_v1',
    'HLT_Mu0_Barrel_L1HP6_IP6_v4',
    'HLT_Mu0_Barrel_L1HP6_v4',
    'HLT_Mu0_Barrel_L1HP7_v4',
    'HLT_Mu0_Barrel_L1HP8_v5',
    'HLT_Mu0_Barrel_L1HP9_v5',
    'HLT_Mu0_Barrel_v7',
    'HLT_Mu10_Barrel_L1HP11_IP4_v1',
    'HLT_Mu10_Barrel_L1HP11_IP6_v7',
    'HLT_Mu12_Barrel_L1HP13_IP4_v1',
    'HLT_Mu12_Barrel_L1HP13_IP6_v1',
    'HLT_Mu4_Barrel_IP4_v1',
    'HLT_Mu4_Barrel_IP6_v1',
    'HLT_Mu6_Barrel_L1HP7_IP6_v4',
    'HLT_Mu7_Barrel_L1HP8_IP6_v5',
    'HLT_Mu8_Barrel_L1HP9_IP6_v5',
    'HLT_Mu9_Barrel_L1HP10_IP6_v7'
)


# stream ParkingSingleMuon13

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingSingleMuon13_datasetParkingSingleMuon13_selector
streamParkingSingleMuon13_datasetParkingSingleMuon13_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingSingleMuon13_datasetParkingSingleMuon13_selector.l1tResults = cms.InputTag('')
streamParkingSingleMuon13_datasetParkingSingleMuon13_selector.throw      = cms.bool(False)
streamParkingSingleMuon13_datasetParkingSingleMuon13_selector.triggerConditions = cms.vstring(
    'HLT_Mu0_Barrel_L1HP10_v7',
    'HLT_Mu0_Barrel_L1HP11_v7',
    'HLT_Mu0_Barrel_L1HP13_v1',
    'HLT_Mu0_Barrel_L1HP6_IP6_v4',
    'HLT_Mu0_Barrel_L1HP6_v4',
    'HLT_Mu0_Barrel_L1HP7_v4',
    'HLT_Mu0_Barrel_L1HP8_v5',
    'HLT_Mu0_Barrel_L1HP9_v5',
    'HLT_Mu0_Barrel_v7',
    'HLT_Mu10_Barrel_L1HP11_IP4_v1',
    'HLT_Mu10_Barrel_L1HP11_IP6_v7',
    'HLT_Mu12_Barrel_L1HP13_IP4_v1',
    'HLT_Mu12_Barrel_L1HP13_IP6_v1',
    'HLT_Mu4_Barrel_IP4_v1',
    'HLT_Mu4_Barrel_IP6_v1',
    'HLT_Mu6_Barrel_L1HP7_IP6_v4',
    'HLT_Mu7_Barrel_L1HP8_IP6_v5',
    'HLT_Mu8_Barrel_L1HP9_IP6_v5',
    'HLT_Mu9_Barrel_L1HP10_IP6_v7'
)


# stream ParkingSingleMuon14

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingSingleMuon14_datasetParkingSingleMuon14_selector
streamParkingSingleMuon14_datasetParkingSingleMuon14_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingSingleMuon14_datasetParkingSingleMuon14_selector.l1tResults = cms.InputTag('')
streamParkingSingleMuon14_datasetParkingSingleMuon14_selector.throw      = cms.bool(False)
streamParkingSingleMuon14_datasetParkingSingleMuon14_selector.triggerConditions = cms.vstring(
    'HLT_Mu0_Barrel_L1HP10_v7',
    'HLT_Mu0_Barrel_L1HP11_v7',
    'HLT_Mu0_Barrel_L1HP13_v1',
    'HLT_Mu0_Barrel_L1HP6_IP6_v4',
    'HLT_Mu0_Barrel_L1HP6_v4',
    'HLT_Mu0_Barrel_L1HP7_v4',
    'HLT_Mu0_Barrel_L1HP8_v5',
    'HLT_Mu0_Barrel_L1HP9_v5',
    'HLT_Mu0_Barrel_v7',
    'HLT_Mu10_Barrel_L1HP11_IP4_v1',
    'HLT_Mu10_Barrel_L1HP11_IP6_v7',
    'HLT_Mu12_Barrel_L1HP13_IP4_v1',
    'HLT_Mu12_Barrel_L1HP13_IP6_v1',
    'HLT_Mu4_Barrel_IP4_v1',
    'HLT_Mu4_Barrel_IP6_v1',
    'HLT_Mu6_Barrel_L1HP7_IP6_v4',
    'HLT_Mu7_Barrel_L1HP8_IP6_v5',
    'HLT_Mu8_Barrel_L1HP9_IP6_v5',
    'HLT_Mu9_Barrel_L1HP10_IP6_v7'
)


# stream ParkingSingleMuon15

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingSingleMuon15_datasetParkingSingleMuon15_selector
streamParkingSingleMuon15_datasetParkingSingleMuon15_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingSingleMuon15_datasetParkingSingleMuon15_selector.l1tResults = cms.InputTag('')
streamParkingSingleMuon15_datasetParkingSingleMuon15_selector.throw      = cms.bool(False)
streamParkingSingleMuon15_datasetParkingSingleMuon15_selector.triggerConditions = cms.vstring(
    'HLT_Mu0_Barrel_L1HP10_v7',
    'HLT_Mu0_Barrel_L1HP11_v7',
    'HLT_Mu0_Barrel_L1HP13_v1',
    'HLT_Mu0_Barrel_L1HP6_IP6_v4',
    'HLT_Mu0_Barrel_L1HP6_v4',
    'HLT_Mu0_Barrel_L1HP7_v4',
    'HLT_Mu0_Barrel_L1HP8_v5',
    'HLT_Mu0_Barrel_L1HP9_v5',
    'HLT_Mu0_Barrel_v7',
    'HLT_Mu10_Barrel_L1HP11_IP4_v1',
    'HLT_Mu10_Barrel_L1HP11_IP6_v7',
    'HLT_Mu12_Barrel_L1HP13_IP4_v1',
    'HLT_Mu12_Barrel_L1HP13_IP6_v1',
    'HLT_Mu4_Barrel_IP4_v1',
    'HLT_Mu4_Barrel_IP6_v1',
    'HLT_Mu6_Barrel_L1HP7_IP6_v4',
    'HLT_Mu7_Barrel_L1HP8_IP6_v5',
    'HLT_Mu8_Barrel_L1HP9_IP6_v5',
    'HLT_Mu9_Barrel_L1HP10_IP6_v7'
)


# stream ParkingSingleMuon2

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingSingleMuon2_datasetParkingSingleMuon2_selector
streamParkingSingleMuon2_datasetParkingSingleMuon2_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingSingleMuon2_datasetParkingSingleMuon2_selector.l1tResults = cms.InputTag('')
streamParkingSingleMuon2_datasetParkingSingleMuon2_selector.throw      = cms.bool(False)
streamParkingSingleMuon2_datasetParkingSingleMuon2_selector.triggerConditions = cms.vstring(
    'HLT_Mu0_Barrel_L1HP10_v7',
    'HLT_Mu0_Barrel_L1HP11_v7',
    'HLT_Mu0_Barrel_L1HP13_v1',
    'HLT_Mu0_Barrel_L1HP6_IP6_v4',
    'HLT_Mu0_Barrel_L1HP6_v4',
    'HLT_Mu0_Barrel_L1HP7_v4',
    'HLT_Mu0_Barrel_L1HP8_v5',
    'HLT_Mu0_Barrel_L1HP9_v5',
    'HLT_Mu0_Barrel_v7',
    'HLT_Mu10_Barrel_L1HP11_IP4_v1',
    'HLT_Mu10_Barrel_L1HP11_IP6_v7',
    'HLT_Mu12_Barrel_L1HP13_IP4_v1',
    'HLT_Mu12_Barrel_L1HP13_IP6_v1',
    'HLT_Mu4_Barrel_IP4_v1',
    'HLT_Mu4_Barrel_IP6_v1',
    'HLT_Mu6_Barrel_L1HP7_IP6_v4',
    'HLT_Mu7_Barrel_L1HP8_IP6_v5',
    'HLT_Mu8_Barrel_L1HP9_IP6_v5',
    'HLT_Mu9_Barrel_L1HP10_IP6_v7'
)


# stream ParkingSingleMuon3

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingSingleMuon3_datasetParkingSingleMuon3_selector
streamParkingSingleMuon3_datasetParkingSingleMuon3_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingSingleMuon3_datasetParkingSingleMuon3_selector.l1tResults = cms.InputTag('')
streamParkingSingleMuon3_datasetParkingSingleMuon3_selector.throw      = cms.bool(False)
streamParkingSingleMuon3_datasetParkingSingleMuon3_selector.triggerConditions = cms.vstring(
    'HLT_Mu0_Barrel_L1HP10_v7',
    'HLT_Mu0_Barrel_L1HP11_v7',
    'HLT_Mu0_Barrel_L1HP13_v1',
    'HLT_Mu0_Barrel_L1HP6_IP6_v4',
    'HLT_Mu0_Barrel_L1HP6_v4',
    'HLT_Mu0_Barrel_L1HP7_v4',
    'HLT_Mu0_Barrel_L1HP8_v5',
    'HLT_Mu0_Barrel_L1HP9_v5',
    'HLT_Mu0_Barrel_v7',
    'HLT_Mu10_Barrel_L1HP11_IP4_v1',
    'HLT_Mu10_Barrel_L1HP11_IP6_v7',
    'HLT_Mu12_Barrel_L1HP13_IP4_v1',
    'HLT_Mu12_Barrel_L1HP13_IP6_v1',
    'HLT_Mu4_Barrel_IP4_v1',
    'HLT_Mu4_Barrel_IP6_v1',
    'HLT_Mu6_Barrel_L1HP7_IP6_v4',
    'HLT_Mu7_Barrel_L1HP8_IP6_v5',
    'HLT_Mu8_Barrel_L1HP9_IP6_v5',
    'HLT_Mu9_Barrel_L1HP10_IP6_v7'
)


# stream ParkingSingleMuon4

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingSingleMuon4_datasetParkingSingleMuon4_selector
streamParkingSingleMuon4_datasetParkingSingleMuon4_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingSingleMuon4_datasetParkingSingleMuon4_selector.l1tResults = cms.InputTag('')
streamParkingSingleMuon4_datasetParkingSingleMuon4_selector.throw      = cms.bool(False)
streamParkingSingleMuon4_datasetParkingSingleMuon4_selector.triggerConditions = cms.vstring(
    'HLT_Mu0_Barrel_L1HP10_v7',
    'HLT_Mu0_Barrel_L1HP11_v7',
    'HLT_Mu0_Barrel_L1HP13_v1',
    'HLT_Mu0_Barrel_L1HP6_IP6_v4',
    'HLT_Mu0_Barrel_L1HP6_v4',
    'HLT_Mu0_Barrel_L1HP7_v4',
    'HLT_Mu0_Barrel_L1HP8_v5',
    'HLT_Mu0_Barrel_L1HP9_v5',
    'HLT_Mu0_Barrel_v7',
    'HLT_Mu10_Barrel_L1HP11_IP4_v1',
    'HLT_Mu10_Barrel_L1HP11_IP6_v7',
    'HLT_Mu12_Barrel_L1HP13_IP4_v1',
    'HLT_Mu12_Barrel_L1HP13_IP6_v1',
    'HLT_Mu4_Barrel_IP4_v1',
    'HLT_Mu4_Barrel_IP6_v1',
    'HLT_Mu6_Barrel_L1HP7_IP6_v4',
    'HLT_Mu7_Barrel_L1HP8_IP6_v5',
    'HLT_Mu8_Barrel_L1HP9_IP6_v5',
    'HLT_Mu9_Barrel_L1HP10_IP6_v7'
)


# stream ParkingSingleMuon5

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingSingleMuon5_datasetParkingSingleMuon5_selector
streamParkingSingleMuon5_datasetParkingSingleMuon5_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingSingleMuon5_datasetParkingSingleMuon5_selector.l1tResults = cms.InputTag('')
streamParkingSingleMuon5_datasetParkingSingleMuon5_selector.throw      = cms.bool(False)
streamParkingSingleMuon5_datasetParkingSingleMuon5_selector.triggerConditions = cms.vstring(
    'HLT_Mu0_Barrel_L1HP10_v7',
    'HLT_Mu0_Barrel_L1HP11_v7',
    'HLT_Mu0_Barrel_L1HP13_v1',
    'HLT_Mu0_Barrel_L1HP6_IP6_v4',
    'HLT_Mu0_Barrel_L1HP6_v4',
    'HLT_Mu0_Barrel_L1HP7_v4',
    'HLT_Mu0_Barrel_L1HP8_v5',
    'HLT_Mu0_Barrel_L1HP9_v5',
    'HLT_Mu0_Barrel_v7',
    'HLT_Mu10_Barrel_L1HP11_IP4_v1',
    'HLT_Mu10_Barrel_L1HP11_IP6_v7',
    'HLT_Mu12_Barrel_L1HP13_IP4_v1',
    'HLT_Mu12_Barrel_L1HP13_IP6_v1',
    'HLT_Mu4_Barrel_IP4_v1',
    'HLT_Mu4_Barrel_IP6_v1',
    'HLT_Mu6_Barrel_L1HP7_IP6_v4',
    'HLT_Mu7_Barrel_L1HP8_IP6_v5',
    'HLT_Mu8_Barrel_L1HP9_IP6_v5',
    'HLT_Mu9_Barrel_L1HP10_IP6_v7'
)


# stream ParkingSingleMuon6

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingSingleMuon6_datasetParkingSingleMuon6_selector
streamParkingSingleMuon6_datasetParkingSingleMuon6_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingSingleMuon6_datasetParkingSingleMuon6_selector.l1tResults = cms.InputTag('')
streamParkingSingleMuon6_datasetParkingSingleMuon6_selector.throw      = cms.bool(False)
streamParkingSingleMuon6_datasetParkingSingleMuon6_selector.triggerConditions = cms.vstring(
    'HLT_Mu0_Barrel_L1HP10_v7',
    'HLT_Mu0_Barrel_L1HP11_v7',
    'HLT_Mu0_Barrel_L1HP13_v1',
    'HLT_Mu0_Barrel_L1HP6_IP6_v4',
    'HLT_Mu0_Barrel_L1HP6_v4',
    'HLT_Mu0_Barrel_L1HP7_v4',
    'HLT_Mu0_Barrel_L1HP8_v5',
    'HLT_Mu0_Barrel_L1HP9_v5',
    'HLT_Mu0_Barrel_v7',
    'HLT_Mu10_Barrel_L1HP11_IP4_v1',
    'HLT_Mu10_Barrel_L1HP11_IP6_v7',
    'HLT_Mu12_Barrel_L1HP13_IP4_v1',
    'HLT_Mu12_Barrel_L1HP13_IP6_v1',
    'HLT_Mu4_Barrel_IP4_v1',
    'HLT_Mu4_Barrel_IP6_v1',
    'HLT_Mu6_Barrel_L1HP7_IP6_v4',
    'HLT_Mu7_Barrel_L1HP8_IP6_v5',
    'HLT_Mu8_Barrel_L1HP9_IP6_v5',
    'HLT_Mu9_Barrel_L1HP10_IP6_v7'
)


# stream ParkingSingleMuon7

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingSingleMuon7_datasetParkingSingleMuon7_selector
streamParkingSingleMuon7_datasetParkingSingleMuon7_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingSingleMuon7_datasetParkingSingleMuon7_selector.l1tResults = cms.InputTag('')
streamParkingSingleMuon7_datasetParkingSingleMuon7_selector.throw      = cms.bool(False)
streamParkingSingleMuon7_datasetParkingSingleMuon7_selector.triggerConditions = cms.vstring(
    'HLT_Mu0_Barrel_L1HP10_v7',
    'HLT_Mu0_Barrel_L1HP11_v7',
    'HLT_Mu0_Barrel_L1HP13_v1',
    'HLT_Mu0_Barrel_L1HP6_IP6_v4',
    'HLT_Mu0_Barrel_L1HP6_v4',
    'HLT_Mu0_Barrel_L1HP7_v4',
    'HLT_Mu0_Barrel_L1HP8_v5',
    'HLT_Mu0_Barrel_L1HP9_v5',
    'HLT_Mu0_Barrel_v7',
    'HLT_Mu10_Barrel_L1HP11_IP4_v1',
    'HLT_Mu10_Barrel_L1HP11_IP6_v7',
    'HLT_Mu12_Barrel_L1HP13_IP4_v1',
    'HLT_Mu12_Barrel_L1HP13_IP6_v1',
    'HLT_Mu4_Barrel_IP4_v1',
    'HLT_Mu4_Barrel_IP6_v1',
    'HLT_Mu6_Barrel_L1HP7_IP6_v4',
    'HLT_Mu7_Barrel_L1HP8_IP6_v5',
    'HLT_Mu8_Barrel_L1HP9_IP6_v5',
    'HLT_Mu9_Barrel_L1HP10_IP6_v7'
)


# stream ParkingSingleMuon8

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingSingleMuon8_datasetParkingSingleMuon8_selector
streamParkingSingleMuon8_datasetParkingSingleMuon8_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingSingleMuon8_datasetParkingSingleMuon8_selector.l1tResults = cms.InputTag('')
streamParkingSingleMuon8_datasetParkingSingleMuon8_selector.throw      = cms.bool(False)
streamParkingSingleMuon8_datasetParkingSingleMuon8_selector.triggerConditions = cms.vstring(
    'HLT_Mu0_Barrel_L1HP10_v7',
    'HLT_Mu0_Barrel_L1HP11_v7',
    'HLT_Mu0_Barrel_L1HP13_v1',
    'HLT_Mu0_Barrel_L1HP6_IP6_v4',
    'HLT_Mu0_Barrel_L1HP6_v4',
    'HLT_Mu0_Barrel_L1HP7_v4',
    'HLT_Mu0_Barrel_L1HP8_v5',
    'HLT_Mu0_Barrel_L1HP9_v5',
    'HLT_Mu0_Barrel_v7',
    'HLT_Mu10_Barrel_L1HP11_IP4_v1',
    'HLT_Mu10_Barrel_L1HP11_IP6_v7',
    'HLT_Mu12_Barrel_L1HP13_IP4_v1',
    'HLT_Mu12_Barrel_L1HP13_IP6_v1',
    'HLT_Mu4_Barrel_IP4_v1',
    'HLT_Mu4_Barrel_IP6_v1',
    'HLT_Mu6_Barrel_L1HP7_IP6_v4',
    'HLT_Mu7_Barrel_L1HP8_IP6_v5',
    'HLT_Mu8_Barrel_L1HP9_IP6_v5',
    'HLT_Mu9_Barrel_L1HP10_IP6_v7'
)


# stream ParkingSingleMuon9

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingSingleMuon9_datasetParkingSingleMuon9_selector
streamParkingSingleMuon9_datasetParkingSingleMuon9_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingSingleMuon9_datasetParkingSingleMuon9_selector.l1tResults = cms.InputTag('')
streamParkingSingleMuon9_datasetParkingSingleMuon9_selector.throw      = cms.bool(False)
streamParkingSingleMuon9_datasetParkingSingleMuon9_selector.triggerConditions = cms.vstring(
    'HLT_Mu0_Barrel_L1HP10_v7',
    'HLT_Mu0_Barrel_L1HP11_v7',
    'HLT_Mu0_Barrel_L1HP13_v1',
    'HLT_Mu0_Barrel_L1HP6_IP6_v4',
    'HLT_Mu0_Barrel_L1HP6_v4',
    'HLT_Mu0_Barrel_L1HP7_v4',
    'HLT_Mu0_Barrel_L1HP8_v5',
    'HLT_Mu0_Barrel_L1HP9_v5',
    'HLT_Mu0_Barrel_v7',
    'HLT_Mu10_Barrel_L1HP11_IP4_v1',
    'HLT_Mu10_Barrel_L1HP11_IP6_v7',
    'HLT_Mu12_Barrel_L1HP13_IP4_v1',
    'HLT_Mu12_Barrel_L1HP13_IP6_v1',
    'HLT_Mu4_Barrel_IP4_v1',
    'HLT_Mu4_Barrel_IP6_v1',
    'HLT_Mu6_Barrel_L1HP7_IP6_v4',
    'HLT_Mu7_Barrel_L1HP8_IP6_v5',
    'HLT_Mu8_Barrel_L1HP9_IP6_v5',
    'HLT_Mu9_Barrel_L1HP10_IP6_v7'
)


# stream ParkingVBF0

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingVBF0_datasetParkingVBF0_selector
streamParkingVBF0_datasetParkingVBF0_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingVBF0_datasetParkingVBF0_selector.l1tResults = cms.InputTag('')
streamParkingVBF0_datasetParkingVBF0_selector.throw      = cms.bool(False)
streamParkingVBF0_datasetParkingVBF0_selector.triggerConditions = cms.vstring(
    'HLT_QuadPFJet100_88_70_30_PNet1CvsAll0p5_VBF3Tight_v11',
    'HLT_QuadPFJet100_88_70_30_v11',
    'HLT_QuadPFJet103_88_75_15_PNet2BTag_0p4_0p12_VBF1_v7',
    'HLT_QuadPFJet103_88_75_15_PNetBTag_0p4_VBF2_v7',
    'HLT_QuadPFJet103_88_75_15_v18',
    'HLT_QuadPFJet105_88_75_30_PNet1CvsAll0p5_VBF3Tight_v11',
    'HLT_QuadPFJet105_88_75_30_v10',
    'HLT_QuadPFJet105_88_76_15_PNet2BTag_0p4_0p12_VBF1_v7',
    'HLT_QuadPFJet105_88_76_15_PNetBTag_0p4_VBF2_v7',
    'HLT_QuadPFJet105_88_76_15_v18',
    'HLT_QuadPFJet111_90_80_15_PNet2BTag_0p4_0p12_VBF1_v7',
    'HLT_QuadPFJet111_90_80_15_PNetBTag_0p4_VBF2_v7',
    'HLT_QuadPFJet111_90_80_15_v18',
    'HLT_QuadPFJet111_90_80_30_PNet1CvsAll0p6_VBF3Tight_v11',
    'HLT_QuadPFJet111_90_80_30_v10',
    'HLT_VBF_DiPFJet115_40_Mjj1000_DoublePNetTauhPFJet20_eta2p3_v1',
    'HLT_VBF_DiPFJet115_40_Mjj1100_DoublePNetTauhPFJet20_eta2p3_v1',
    'HLT_VBF_DiPFJet115_40_Mjj1200_DoublePNetTauhPFJet20_eta2p3_v1',
    'HLT_VBF_DiPFJet115_40_Mjj850_DoublePNetTauhPFJet20_eta2p3_v1',
    'HLT_VBF_DiPFJet125_45_Mjj1050_v9',
    'HLT_VBF_DiPFJet125_45_Mjj1150_v1',
    'HLT_VBF_DiPFJet125_45_Mjj1250_v1',
    'HLT_VBF_DiPFJet45_Mjj650_PNetTauhPFJet45_L2NN_eta2p3_v7',
    'HLT_VBF_DiPFJet45_Mjj750_PNetTauhPFJet45_L2NN_eta2p3_v7',
    'HLT_VBF_DiPFJet45_Mjj850_PNetTauhPFJet45_L2NN_eta2p3_v1',
    'HLT_VBF_DiPFJet50_Mjj600_Ele22_eta2p1_WPTight_Gsf_v7',
    'HLT_VBF_DiPFJet50_Mjj650_Photon22_v7',
    'HLT_VBF_DiPFJet50_Mjj700_Ele22_eta2p1_WPTight_Gsf_v1',
    'HLT_VBF_DiPFJet50_Mjj750_Photon22_v7',
    'HLT_VBF_DiPFJet50_Mjj800_Ele22_eta2p1_WPTight_Gsf_v1',
    'HLT_VBF_DiPFJet50_Mjj850_Photon22_v1',
    'HLT_VBF_DiPFJet75_45_Mjj1000_DiPFJet60_v1',
    'HLT_VBF_DiPFJet75_45_Mjj800_DiPFJet60_v7',
    'HLT_VBF_DiPFJet75_45_Mjj900_DiPFJet60_v1',
    'HLT_VBF_DiPFJet80_45_Mjj650_PFMETNoMu85_v7',
    'HLT_VBF_DiPFJet80_45_Mjj750_PFMETNoMu85_v7',
    'HLT_VBF_DiPFJet80_45_Mjj850_PFMETNoMu85_v1',
    'HLT_VBF_DiPFJet95_45_Mjj750_Mu3_TrkIsoVVL_v7',
    'HLT_VBF_DiPFJet95_45_Mjj850_Mu3_TrkIsoVVL_v7',
    'HLT_VBF_DiPFJet95_45_Mjj950_Mu3_TrkIsoVVL_v1'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingVBF0_datasetParkingVBF1_selector
streamParkingVBF0_datasetParkingVBF1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingVBF0_datasetParkingVBF1_selector.l1tResults = cms.InputTag('')
streamParkingVBF0_datasetParkingVBF1_selector.throw      = cms.bool(False)
streamParkingVBF0_datasetParkingVBF1_selector.triggerConditions = cms.vstring(
    'HLT_QuadPFJet100_88_70_30_PNet1CvsAll0p5_VBF3Tight_v11',
    'HLT_QuadPFJet100_88_70_30_v11',
    'HLT_QuadPFJet103_88_75_15_PNet2BTag_0p4_0p12_VBF1_v7',
    'HLT_QuadPFJet103_88_75_15_PNetBTag_0p4_VBF2_v7',
    'HLT_QuadPFJet103_88_75_15_v18',
    'HLT_QuadPFJet105_88_75_30_PNet1CvsAll0p5_VBF3Tight_v11',
    'HLT_QuadPFJet105_88_75_30_v10',
    'HLT_QuadPFJet105_88_76_15_PNet2BTag_0p4_0p12_VBF1_v7',
    'HLT_QuadPFJet105_88_76_15_PNetBTag_0p4_VBF2_v7',
    'HLT_QuadPFJet105_88_76_15_v18',
    'HLT_QuadPFJet111_90_80_15_PNet2BTag_0p4_0p12_VBF1_v7',
    'HLT_QuadPFJet111_90_80_15_PNetBTag_0p4_VBF2_v7',
    'HLT_QuadPFJet111_90_80_15_v18',
    'HLT_QuadPFJet111_90_80_30_PNet1CvsAll0p6_VBF3Tight_v11',
    'HLT_QuadPFJet111_90_80_30_v10',
    'HLT_VBF_DiPFJet115_40_Mjj1000_DoublePNetTauhPFJet20_eta2p3_v1',
    'HLT_VBF_DiPFJet115_40_Mjj1100_DoublePNetTauhPFJet20_eta2p3_v1',
    'HLT_VBF_DiPFJet115_40_Mjj1200_DoublePNetTauhPFJet20_eta2p3_v1',
    'HLT_VBF_DiPFJet115_40_Mjj850_DoublePNetTauhPFJet20_eta2p3_v1',
    'HLT_VBF_DiPFJet125_45_Mjj1050_v9',
    'HLT_VBF_DiPFJet125_45_Mjj1150_v1',
    'HLT_VBF_DiPFJet125_45_Mjj1250_v1',
    'HLT_VBF_DiPFJet45_Mjj650_PNetTauhPFJet45_L2NN_eta2p3_v7',
    'HLT_VBF_DiPFJet45_Mjj750_PNetTauhPFJet45_L2NN_eta2p3_v7',
    'HLT_VBF_DiPFJet45_Mjj850_PNetTauhPFJet45_L2NN_eta2p3_v1',
    'HLT_VBF_DiPFJet50_Mjj600_Ele22_eta2p1_WPTight_Gsf_v7',
    'HLT_VBF_DiPFJet50_Mjj650_Photon22_v7',
    'HLT_VBF_DiPFJet50_Mjj700_Ele22_eta2p1_WPTight_Gsf_v1',
    'HLT_VBF_DiPFJet50_Mjj750_Photon22_v7',
    'HLT_VBF_DiPFJet50_Mjj800_Ele22_eta2p1_WPTight_Gsf_v1',
    'HLT_VBF_DiPFJet50_Mjj850_Photon22_v1',
    'HLT_VBF_DiPFJet75_45_Mjj1000_DiPFJet60_v1',
    'HLT_VBF_DiPFJet75_45_Mjj800_DiPFJet60_v7',
    'HLT_VBF_DiPFJet75_45_Mjj900_DiPFJet60_v1',
    'HLT_VBF_DiPFJet80_45_Mjj650_PFMETNoMu85_v7',
    'HLT_VBF_DiPFJet80_45_Mjj750_PFMETNoMu85_v7',
    'HLT_VBF_DiPFJet80_45_Mjj850_PFMETNoMu85_v1',
    'HLT_VBF_DiPFJet95_45_Mjj750_Mu3_TrkIsoVVL_v7',
    'HLT_VBF_DiPFJet95_45_Mjj850_Mu3_TrkIsoVVL_v7',
    'HLT_VBF_DiPFJet95_45_Mjj950_Mu3_TrkIsoVVL_v1'
)


# stream ParkingVBF1

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingVBF1_datasetParkingVBF2_selector
streamParkingVBF1_datasetParkingVBF2_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingVBF1_datasetParkingVBF2_selector.l1tResults = cms.InputTag('')
streamParkingVBF1_datasetParkingVBF2_selector.throw      = cms.bool(False)
streamParkingVBF1_datasetParkingVBF2_selector.triggerConditions = cms.vstring(
    'HLT_QuadPFJet100_88_70_30_PNet1CvsAll0p5_VBF3Tight_v11',
    'HLT_QuadPFJet100_88_70_30_v11',
    'HLT_QuadPFJet103_88_75_15_PNet2BTag_0p4_0p12_VBF1_v7',
    'HLT_QuadPFJet103_88_75_15_PNetBTag_0p4_VBF2_v7',
    'HLT_QuadPFJet103_88_75_15_v18',
    'HLT_QuadPFJet105_88_75_30_PNet1CvsAll0p5_VBF3Tight_v11',
    'HLT_QuadPFJet105_88_75_30_v10',
    'HLT_QuadPFJet105_88_76_15_PNet2BTag_0p4_0p12_VBF1_v7',
    'HLT_QuadPFJet105_88_76_15_PNetBTag_0p4_VBF2_v7',
    'HLT_QuadPFJet105_88_76_15_v18',
    'HLT_QuadPFJet111_90_80_15_PNet2BTag_0p4_0p12_VBF1_v7',
    'HLT_QuadPFJet111_90_80_15_PNetBTag_0p4_VBF2_v7',
    'HLT_QuadPFJet111_90_80_15_v18',
    'HLT_QuadPFJet111_90_80_30_PNet1CvsAll0p6_VBF3Tight_v11',
    'HLT_QuadPFJet111_90_80_30_v10',
    'HLT_VBF_DiPFJet115_40_Mjj1000_DoublePNetTauhPFJet20_eta2p3_v1',
    'HLT_VBF_DiPFJet115_40_Mjj1100_DoublePNetTauhPFJet20_eta2p3_v1',
    'HLT_VBF_DiPFJet115_40_Mjj1200_DoublePNetTauhPFJet20_eta2p3_v1',
    'HLT_VBF_DiPFJet115_40_Mjj850_DoublePNetTauhPFJet20_eta2p3_v1',
    'HLT_VBF_DiPFJet125_45_Mjj1050_v9',
    'HLT_VBF_DiPFJet125_45_Mjj1150_v1',
    'HLT_VBF_DiPFJet125_45_Mjj1250_v1',
    'HLT_VBF_DiPFJet45_Mjj650_PNetTauhPFJet45_L2NN_eta2p3_v7',
    'HLT_VBF_DiPFJet45_Mjj750_PNetTauhPFJet45_L2NN_eta2p3_v7',
    'HLT_VBF_DiPFJet45_Mjj850_PNetTauhPFJet45_L2NN_eta2p3_v1',
    'HLT_VBF_DiPFJet50_Mjj600_Ele22_eta2p1_WPTight_Gsf_v7',
    'HLT_VBF_DiPFJet50_Mjj650_Photon22_v7',
    'HLT_VBF_DiPFJet50_Mjj700_Ele22_eta2p1_WPTight_Gsf_v1',
    'HLT_VBF_DiPFJet50_Mjj750_Photon22_v7',
    'HLT_VBF_DiPFJet50_Mjj800_Ele22_eta2p1_WPTight_Gsf_v1',
    'HLT_VBF_DiPFJet50_Mjj850_Photon22_v1',
    'HLT_VBF_DiPFJet75_45_Mjj1000_DiPFJet60_v1',
    'HLT_VBF_DiPFJet75_45_Mjj800_DiPFJet60_v7',
    'HLT_VBF_DiPFJet75_45_Mjj900_DiPFJet60_v1',
    'HLT_VBF_DiPFJet80_45_Mjj650_PFMETNoMu85_v7',
    'HLT_VBF_DiPFJet80_45_Mjj750_PFMETNoMu85_v7',
    'HLT_VBF_DiPFJet80_45_Mjj850_PFMETNoMu85_v1',
    'HLT_VBF_DiPFJet95_45_Mjj750_Mu3_TrkIsoVVL_v7',
    'HLT_VBF_DiPFJet95_45_Mjj850_Mu3_TrkIsoVVL_v7',
    'HLT_VBF_DiPFJet95_45_Mjj950_Mu3_TrkIsoVVL_v1'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingVBF1_datasetParkingVBF3_selector
streamParkingVBF1_datasetParkingVBF3_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingVBF1_datasetParkingVBF3_selector.l1tResults = cms.InputTag('')
streamParkingVBF1_datasetParkingVBF3_selector.throw      = cms.bool(False)
streamParkingVBF1_datasetParkingVBF3_selector.triggerConditions = cms.vstring(
    'HLT_QuadPFJet100_88_70_30_PNet1CvsAll0p5_VBF3Tight_v11',
    'HLT_QuadPFJet100_88_70_30_v11',
    'HLT_QuadPFJet103_88_75_15_PNet2BTag_0p4_0p12_VBF1_v7',
    'HLT_QuadPFJet103_88_75_15_PNetBTag_0p4_VBF2_v7',
    'HLT_QuadPFJet103_88_75_15_v18',
    'HLT_QuadPFJet105_88_75_30_PNet1CvsAll0p5_VBF3Tight_v11',
    'HLT_QuadPFJet105_88_75_30_v10',
    'HLT_QuadPFJet105_88_76_15_PNet2BTag_0p4_0p12_VBF1_v7',
    'HLT_QuadPFJet105_88_76_15_PNetBTag_0p4_VBF2_v7',
    'HLT_QuadPFJet105_88_76_15_v18',
    'HLT_QuadPFJet111_90_80_15_PNet2BTag_0p4_0p12_VBF1_v7',
    'HLT_QuadPFJet111_90_80_15_PNetBTag_0p4_VBF2_v7',
    'HLT_QuadPFJet111_90_80_15_v18',
    'HLT_QuadPFJet111_90_80_30_PNet1CvsAll0p6_VBF3Tight_v11',
    'HLT_QuadPFJet111_90_80_30_v10',
    'HLT_VBF_DiPFJet115_40_Mjj1000_DoublePNetTauhPFJet20_eta2p3_v1',
    'HLT_VBF_DiPFJet115_40_Mjj1100_DoublePNetTauhPFJet20_eta2p3_v1',
    'HLT_VBF_DiPFJet115_40_Mjj1200_DoublePNetTauhPFJet20_eta2p3_v1',
    'HLT_VBF_DiPFJet115_40_Mjj850_DoublePNetTauhPFJet20_eta2p3_v1',
    'HLT_VBF_DiPFJet125_45_Mjj1050_v9',
    'HLT_VBF_DiPFJet125_45_Mjj1150_v1',
    'HLT_VBF_DiPFJet125_45_Mjj1250_v1',
    'HLT_VBF_DiPFJet45_Mjj650_PNetTauhPFJet45_L2NN_eta2p3_v7',
    'HLT_VBF_DiPFJet45_Mjj750_PNetTauhPFJet45_L2NN_eta2p3_v7',
    'HLT_VBF_DiPFJet45_Mjj850_PNetTauhPFJet45_L2NN_eta2p3_v1',
    'HLT_VBF_DiPFJet50_Mjj600_Ele22_eta2p1_WPTight_Gsf_v7',
    'HLT_VBF_DiPFJet50_Mjj650_Photon22_v7',
    'HLT_VBF_DiPFJet50_Mjj700_Ele22_eta2p1_WPTight_Gsf_v1',
    'HLT_VBF_DiPFJet50_Mjj750_Photon22_v7',
    'HLT_VBF_DiPFJet50_Mjj800_Ele22_eta2p1_WPTight_Gsf_v1',
    'HLT_VBF_DiPFJet50_Mjj850_Photon22_v1',
    'HLT_VBF_DiPFJet75_45_Mjj1000_DiPFJet60_v1',
    'HLT_VBF_DiPFJet75_45_Mjj800_DiPFJet60_v7',
    'HLT_VBF_DiPFJet75_45_Mjj900_DiPFJet60_v1',
    'HLT_VBF_DiPFJet80_45_Mjj650_PFMETNoMu85_v7',
    'HLT_VBF_DiPFJet80_45_Mjj750_PFMETNoMu85_v7',
    'HLT_VBF_DiPFJet80_45_Mjj850_PFMETNoMu85_v1',
    'HLT_VBF_DiPFJet95_45_Mjj750_Mu3_TrkIsoVVL_v7',
    'HLT_VBF_DiPFJet95_45_Mjj850_Mu3_TrkIsoVVL_v7',
    'HLT_VBF_DiPFJet95_45_Mjj950_Mu3_TrkIsoVVL_v1'
)


# stream ParkingVBF2

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingVBF2_datasetParkingVBF4_selector
streamParkingVBF2_datasetParkingVBF4_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingVBF2_datasetParkingVBF4_selector.l1tResults = cms.InputTag('')
streamParkingVBF2_datasetParkingVBF4_selector.throw      = cms.bool(False)
streamParkingVBF2_datasetParkingVBF4_selector.triggerConditions = cms.vstring(
    'HLT_QuadPFJet100_88_70_30_PNet1CvsAll0p5_VBF3Tight_v11',
    'HLT_QuadPFJet100_88_70_30_v11',
    'HLT_QuadPFJet103_88_75_15_PNet2BTag_0p4_0p12_VBF1_v7',
    'HLT_QuadPFJet103_88_75_15_PNetBTag_0p4_VBF2_v7',
    'HLT_QuadPFJet103_88_75_15_v18',
    'HLT_QuadPFJet105_88_75_30_PNet1CvsAll0p5_VBF3Tight_v11',
    'HLT_QuadPFJet105_88_75_30_v10',
    'HLT_QuadPFJet105_88_76_15_PNet2BTag_0p4_0p12_VBF1_v7',
    'HLT_QuadPFJet105_88_76_15_PNetBTag_0p4_VBF2_v7',
    'HLT_QuadPFJet105_88_76_15_v18',
    'HLT_QuadPFJet111_90_80_15_PNet2BTag_0p4_0p12_VBF1_v7',
    'HLT_QuadPFJet111_90_80_15_PNetBTag_0p4_VBF2_v7',
    'HLT_QuadPFJet111_90_80_15_v18',
    'HLT_QuadPFJet111_90_80_30_PNet1CvsAll0p6_VBF3Tight_v11',
    'HLT_QuadPFJet111_90_80_30_v10',
    'HLT_VBF_DiPFJet115_40_Mjj1000_DoublePNetTauhPFJet20_eta2p3_v1',
    'HLT_VBF_DiPFJet115_40_Mjj1100_DoublePNetTauhPFJet20_eta2p3_v1',
    'HLT_VBF_DiPFJet115_40_Mjj1200_DoublePNetTauhPFJet20_eta2p3_v1',
    'HLT_VBF_DiPFJet115_40_Mjj850_DoublePNetTauhPFJet20_eta2p3_v1',
    'HLT_VBF_DiPFJet125_45_Mjj1050_v9',
    'HLT_VBF_DiPFJet125_45_Mjj1150_v1',
    'HLT_VBF_DiPFJet125_45_Mjj1250_v1',
    'HLT_VBF_DiPFJet45_Mjj650_PNetTauhPFJet45_L2NN_eta2p3_v7',
    'HLT_VBF_DiPFJet45_Mjj750_PNetTauhPFJet45_L2NN_eta2p3_v7',
    'HLT_VBF_DiPFJet45_Mjj850_PNetTauhPFJet45_L2NN_eta2p3_v1',
    'HLT_VBF_DiPFJet50_Mjj600_Ele22_eta2p1_WPTight_Gsf_v7',
    'HLT_VBF_DiPFJet50_Mjj650_Photon22_v7',
    'HLT_VBF_DiPFJet50_Mjj700_Ele22_eta2p1_WPTight_Gsf_v1',
    'HLT_VBF_DiPFJet50_Mjj750_Photon22_v7',
    'HLT_VBF_DiPFJet50_Mjj800_Ele22_eta2p1_WPTight_Gsf_v1',
    'HLT_VBF_DiPFJet50_Mjj850_Photon22_v1',
    'HLT_VBF_DiPFJet75_45_Mjj1000_DiPFJet60_v1',
    'HLT_VBF_DiPFJet75_45_Mjj800_DiPFJet60_v7',
    'HLT_VBF_DiPFJet75_45_Mjj900_DiPFJet60_v1',
    'HLT_VBF_DiPFJet80_45_Mjj650_PFMETNoMu85_v7',
    'HLT_VBF_DiPFJet80_45_Mjj750_PFMETNoMu85_v7',
    'HLT_VBF_DiPFJet80_45_Mjj850_PFMETNoMu85_v1',
    'HLT_VBF_DiPFJet95_45_Mjj750_Mu3_TrkIsoVVL_v7',
    'HLT_VBF_DiPFJet95_45_Mjj850_Mu3_TrkIsoVVL_v7',
    'HLT_VBF_DiPFJet95_45_Mjj950_Mu3_TrkIsoVVL_v1'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingVBF2_datasetParkingVBF5_selector
streamParkingVBF2_datasetParkingVBF5_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingVBF2_datasetParkingVBF5_selector.l1tResults = cms.InputTag('')
streamParkingVBF2_datasetParkingVBF5_selector.throw      = cms.bool(False)
streamParkingVBF2_datasetParkingVBF5_selector.triggerConditions = cms.vstring(
    'HLT_QuadPFJet100_88_70_30_PNet1CvsAll0p5_VBF3Tight_v11',
    'HLT_QuadPFJet100_88_70_30_v11',
    'HLT_QuadPFJet103_88_75_15_PNet2BTag_0p4_0p12_VBF1_v7',
    'HLT_QuadPFJet103_88_75_15_PNetBTag_0p4_VBF2_v7',
    'HLT_QuadPFJet103_88_75_15_v18',
    'HLT_QuadPFJet105_88_75_30_PNet1CvsAll0p5_VBF3Tight_v11',
    'HLT_QuadPFJet105_88_75_30_v10',
    'HLT_QuadPFJet105_88_76_15_PNet2BTag_0p4_0p12_VBF1_v7',
    'HLT_QuadPFJet105_88_76_15_PNetBTag_0p4_VBF2_v7',
    'HLT_QuadPFJet105_88_76_15_v18',
    'HLT_QuadPFJet111_90_80_15_PNet2BTag_0p4_0p12_VBF1_v7',
    'HLT_QuadPFJet111_90_80_15_PNetBTag_0p4_VBF2_v7',
    'HLT_QuadPFJet111_90_80_15_v18',
    'HLT_QuadPFJet111_90_80_30_PNet1CvsAll0p6_VBF3Tight_v11',
    'HLT_QuadPFJet111_90_80_30_v10',
    'HLT_VBF_DiPFJet115_40_Mjj1000_DoublePNetTauhPFJet20_eta2p3_v1',
    'HLT_VBF_DiPFJet115_40_Mjj1100_DoublePNetTauhPFJet20_eta2p3_v1',
    'HLT_VBF_DiPFJet115_40_Mjj1200_DoublePNetTauhPFJet20_eta2p3_v1',
    'HLT_VBF_DiPFJet115_40_Mjj850_DoublePNetTauhPFJet20_eta2p3_v1',
    'HLT_VBF_DiPFJet125_45_Mjj1050_v9',
    'HLT_VBF_DiPFJet125_45_Mjj1150_v1',
    'HLT_VBF_DiPFJet125_45_Mjj1250_v1',
    'HLT_VBF_DiPFJet45_Mjj650_PNetTauhPFJet45_L2NN_eta2p3_v7',
    'HLT_VBF_DiPFJet45_Mjj750_PNetTauhPFJet45_L2NN_eta2p3_v7',
    'HLT_VBF_DiPFJet45_Mjj850_PNetTauhPFJet45_L2NN_eta2p3_v1',
    'HLT_VBF_DiPFJet50_Mjj600_Ele22_eta2p1_WPTight_Gsf_v7',
    'HLT_VBF_DiPFJet50_Mjj650_Photon22_v7',
    'HLT_VBF_DiPFJet50_Mjj700_Ele22_eta2p1_WPTight_Gsf_v1',
    'HLT_VBF_DiPFJet50_Mjj750_Photon22_v7',
    'HLT_VBF_DiPFJet50_Mjj800_Ele22_eta2p1_WPTight_Gsf_v1',
    'HLT_VBF_DiPFJet50_Mjj850_Photon22_v1',
    'HLT_VBF_DiPFJet75_45_Mjj1000_DiPFJet60_v1',
    'HLT_VBF_DiPFJet75_45_Mjj800_DiPFJet60_v7',
    'HLT_VBF_DiPFJet75_45_Mjj900_DiPFJet60_v1',
    'HLT_VBF_DiPFJet80_45_Mjj650_PFMETNoMu85_v7',
    'HLT_VBF_DiPFJet80_45_Mjj750_PFMETNoMu85_v7',
    'HLT_VBF_DiPFJet80_45_Mjj850_PFMETNoMu85_v1',
    'HLT_VBF_DiPFJet95_45_Mjj750_Mu3_TrkIsoVVL_v7',
    'HLT_VBF_DiPFJet95_45_Mjj850_Mu3_TrkIsoVVL_v7',
    'HLT_VBF_DiPFJet95_45_Mjj950_Mu3_TrkIsoVVL_v1'
)


# stream ParkingVBF3

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingVBF3_datasetParkingVBF6_selector
streamParkingVBF3_datasetParkingVBF6_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingVBF3_datasetParkingVBF6_selector.l1tResults = cms.InputTag('')
streamParkingVBF3_datasetParkingVBF6_selector.throw      = cms.bool(False)
streamParkingVBF3_datasetParkingVBF6_selector.triggerConditions = cms.vstring(
    'HLT_QuadPFJet100_88_70_30_PNet1CvsAll0p5_VBF3Tight_v11',
    'HLT_QuadPFJet100_88_70_30_v11',
    'HLT_QuadPFJet103_88_75_15_PNet2BTag_0p4_0p12_VBF1_v7',
    'HLT_QuadPFJet103_88_75_15_PNetBTag_0p4_VBF2_v7',
    'HLT_QuadPFJet103_88_75_15_v18',
    'HLT_QuadPFJet105_88_75_30_PNet1CvsAll0p5_VBF3Tight_v11',
    'HLT_QuadPFJet105_88_75_30_v10',
    'HLT_QuadPFJet105_88_76_15_PNet2BTag_0p4_0p12_VBF1_v7',
    'HLT_QuadPFJet105_88_76_15_PNetBTag_0p4_VBF2_v7',
    'HLT_QuadPFJet105_88_76_15_v18',
    'HLT_QuadPFJet111_90_80_15_PNet2BTag_0p4_0p12_VBF1_v7',
    'HLT_QuadPFJet111_90_80_15_PNetBTag_0p4_VBF2_v7',
    'HLT_QuadPFJet111_90_80_15_v18',
    'HLT_QuadPFJet111_90_80_30_PNet1CvsAll0p6_VBF3Tight_v11',
    'HLT_QuadPFJet111_90_80_30_v10',
    'HLT_VBF_DiPFJet115_40_Mjj1000_DoublePNetTauhPFJet20_eta2p3_v1',
    'HLT_VBF_DiPFJet115_40_Mjj1100_DoublePNetTauhPFJet20_eta2p3_v1',
    'HLT_VBF_DiPFJet115_40_Mjj1200_DoublePNetTauhPFJet20_eta2p3_v1',
    'HLT_VBF_DiPFJet115_40_Mjj850_DoublePNetTauhPFJet20_eta2p3_v1',
    'HLT_VBF_DiPFJet125_45_Mjj1050_v9',
    'HLT_VBF_DiPFJet125_45_Mjj1150_v1',
    'HLT_VBF_DiPFJet125_45_Mjj1250_v1',
    'HLT_VBF_DiPFJet45_Mjj650_PNetTauhPFJet45_L2NN_eta2p3_v7',
    'HLT_VBF_DiPFJet45_Mjj750_PNetTauhPFJet45_L2NN_eta2p3_v7',
    'HLT_VBF_DiPFJet45_Mjj850_PNetTauhPFJet45_L2NN_eta2p3_v1',
    'HLT_VBF_DiPFJet50_Mjj600_Ele22_eta2p1_WPTight_Gsf_v7',
    'HLT_VBF_DiPFJet50_Mjj650_Photon22_v7',
    'HLT_VBF_DiPFJet50_Mjj700_Ele22_eta2p1_WPTight_Gsf_v1',
    'HLT_VBF_DiPFJet50_Mjj750_Photon22_v7',
    'HLT_VBF_DiPFJet50_Mjj800_Ele22_eta2p1_WPTight_Gsf_v1',
    'HLT_VBF_DiPFJet50_Mjj850_Photon22_v1',
    'HLT_VBF_DiPFJet75_45_Mjj1000_DiPFJet60_v1',
    'HLT_VBF_DiPFJet75_45_Mjj800_DiPFJet60_v7',
    'HLT_VBF_DiPFJet75_45_Mjj900_DiPFJet60_v1',
    'HLT_VBF_DiPFJet80_45_Mjj650_PFMETNoMu85_v7',
    'HLT_VBF_DiPFJet80_45_Mjj750_PFMETNoMu85_v7',
    'HLT_VBF_DiPFJet80_45_Mjj850_PFMETNoMu85_v1',
    'HLT_VBF_DiPFJet95_45_Mjj750_Mu3_TrkIsoVVL_v7',
    'HLT_VBF_DiPFJet95_45_Mjj850_Mu3_TrkIsoVVL_v7',
    'HLT_VBF_DiPFJet95_45_Mjj950_Mu3_TrkIsoVVL_v1'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingVBF3_datasetParkingVBF7_selector
streamParkingVBF3_datasetParkingVBF7_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingVBF3_datasetParkingVBF7_selector.l1tResults = cms.InputTag('')
streamParkingVBF3_datasetParkingVBF7_selector.throw      = cms.bool(False)
streamParkingVBF3_datasetParkingVBF7_selector.triggerConditions = cms.vstring(
    'HLT_QuadPFJet100_88_70_30_PNet1CvsAll0p5_VBF3Tight_v11',
    'HLT_QuadPFJet100_88_70_30_v11',
    'HLT_QuadPFJet103_88_75_15_PNet2BTag_0p4_0p12_VBF1_v7',
    'HLT_QuadPFJet103_88_75_15_PNetBTag_0p4_VBF2_v7',
    'HLT_QuadPFJet103_88_75_15_v18',
    'HLT_QuadPFJet105_88_75_30_PNet1CvsAll0p5_VBF3Tight_v11',
    'HLT_QuadPFJet105_88_75_30_v10',
    'HLT_QuadPFJet105_88_76_15_PNet2BTag_0p4_0p12_VBF1_v7',
    'HLT_QuadPFJet105_88_76_15_PNetBTag_0p4_VBF2_v7',
    'HLT_QuadPFJet105_88_76_15_v18',
    'HLT_QuadPFJet111_90_80_15_PNet2BTag_0p4_0p12_VBF1_v7',
    'HLT_QuadPFJet111_90_80_15_PNetBTag_0p4_VBF2_v7',
    'HLT_QuadPFJet111_90_80_15_v18',
    'HLT_QuadPFJet111_90_80_30_PNet1CvsAll0p6_VBF3Tight_v11',
    'HLT_QuadPFJet111_90_80_30_v10',
    'HLT_VBF_DiPFJet115_40_Mjj1000_DoublePNetTauhPFJet20_eta2p3_v1',
    'HLT_VBF_DiPFJet115_40_Mjj1100_DoublePNetTauhPFJet20_eta2p3_v1',
    'HLT_VBF_DiPFJet115_40_Mjj1200_DoublePNetTauhPFJet20_eta2p3_v1',
    'HLT_VBF_DiPFJet115_40_Mjj850_DoublePNetTauhPFJet20_eta2p3_v1',
    'HLT_VBF_DiPFJet125_45_Mjj1050_v9',
    'HLT_VBF_DiPFJet125_45_Mjj1150_v1',
    'HLT_VBF_DiPFJet125_45_Mjj1250_v1',
    'HLT_VBF_DiPFJet45_Mjj650_PNetTauhPFJet45_L2NN_eta2p3_v7',
    'HLT_VBF_DiPFJet45_Mjj750_PNetTauhPFJet45_L2NN_eta2p3_v7',
    'HLT_VBF_DiPFJet45_Mjj850_PNetTauhPFJet45_L2NN_eta2p3_v1',
    'HLT_VBF_DiPFJet50_Mjj600_Ele22_eta2p1_WPTight_Gsf_v7',
    'HLT_VBF_DiPFJet50_Mjj650_Photon22_v7',
    'HLT_VBF_DiPFJet50_Mjj700_Ele22_eta2p1_WPTight_Gsf_v1',
    'HLT_VBF_DiPFJet50_Mjj750_Photon22_v7',
    'HLT_VBF_DiPFJet50_Mjj800_Ele22_eta2p1_WPTight_Gsf_v1',
    'HLT_VBF_DiPFJet50_Mjj850_Photon22_v1',
    'HLT_VBF_DiPFJet75_45_Mjj1000_DiPFJet60_v1',
    'HLT_VBF_DiPFJet75_45_Mjj800_DiPFJet60_v7',
    'HLT_VBF_DiPFJet75_45_Mjj900_DiPFJet60_v1',
    'HLT_VBF_DiPFJet80_45_Mjj650_PFMETNoMu85_v7',
    'HLT_VBF_DiPFJet80_45_Mjj750_PFMETNoMu85_v7',
    'HLT_VBF_DiPFJet80_45_Mjj850_PFMETNoMu85_v1',
    'HLT_VBF_DiPFJet95_45_Mjj750_Mu3_TrkIsoVVL_v7',
    'HLT_VBF_DiPFJet95_45_Mjj850_Mu3_TrkIsoVVL_v7',
    'HLT_VBF_DiPFJet95_45_Mjj950_Mu3_TrkIsoVVL_v1'
)


# stream PhysicsBTagMuEGTau

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsBTagMuEGTau_datasetBTagMu_selector
streamPhysicsBTagMuEGTau_datasetBTagMu_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsBTagMuEGTau_datasetBTagMu_selector.l1tResults = cms.InputTag('')
streamPhysicsBTagMuEGTau_datasetBTagMu_selector.throw      = cms.bool(False)
streamPhysicsBTagMuEGTau_datasetBTagMu_selector.triggerConditions = cms.vstring(
    'HLT_BTagMu_AK4DiJet110_Mu5_v26',
    'HLT_BTagMu_AK4DiJet170_Mu5_v25',
    'HLT_BTagMu_AK4DiJet20_Mu5_v26',
    'HLT_BTagMu_AK4DiJet40_Mu5_v26',
    'HLT_BTagMu_AK4DiJet70_Mu5_v26',
    'HLT_BTagMu_AK4Jet300_Mu5_v25',
    'HLT_BTagMu_AK8DiJet170_Mu5_v22',
    'HLT_BTagMu_AK8Jet170_DoubleMu5_v15',
    'HLT_BTagMu_AK8Jet300_Mu5_v25'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsBTagMuEGTau_datasetMuonEG_selector
streamPhysicsBTagMuEGTau_datasetMuonEG_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsBTagMuEGTau_datasetMuonEG_selector.l1tResults = cms.InputTag('')
streamPhysicsBTagMuEGTau_datasetMuonEG_selector.throw      = cms.bool(False)
streamPhysicsBTagMuEGTau_datasetMuonEG_selector.triggerConditions = cms.vstring(
    'HLT_DiMu4_Ele9_CaloIdL_TrackIdL_DZ_Mass3p8_v30',
    'HLT_DiMu9_Ele9_CaloIdL_TrackIdL_DZ_v30',
    'HLT_DiMu9_Ele9_CaloIdL_TrackIdL_v30',
    'HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_v28',
    'HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_v20',
    'HLT_Mu17_Photon30_IsoCaloId_v19',
    'HLT_Mu20NoFiltersNoVtxDisplaced_Photon20_CaloCustomId_v12',
    'HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v28',
    'HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_v20',
    'HLT_Mu27_Ele37_CaloIdL_MW_v18',
    'HLT_Mu37_Ele27_CaloIdL_MW_v18',
    'HLT_Mu38NoFiltersNoVtxDisplaced_Photon38_CaloIdL_v12',
    'HLT_Mu43NoFiltersNoVtxDisplaced_Photon43_CaloIdL_v12',
    'HLT_Mu43NoFiltersNoVtx_Photon43_CaloIdL_v16',
    'HLT_Mu48NoFiltersNoVtx_Photon48_CaloIdL_v16',
    'HLT_Mu8_DiEle12_CaloIdL_TrackIdL_DZ_v31',
    'HLT_Mu8_DiEle12_CaloIdL_TrackIdL_v31',
    'HLT_Mu8_Ele8_CaloIdM_TrackIdM_Mass8_PFHT350_DZ_v32',
    'HLT_Mu8_Ele8_CaloIdM_TrackIdM_Mass8_PFHT350_v32',
    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_DoubleAK4PFJet60_30_PNet2BTagMean0p50_v11',
    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_DoubleAK4PFJet60_30_v11',
    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_L1HT200_QuadPFJet20_v1',
    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_L1HT200_QuadPFJet25_PNet1BTag0p50_v1',
    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_L1HT200_QuadPFJet25_v1',
    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_L1HT200_v1',
    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFDiJet30_PNet2BTagMean0p50_v10',
    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFDiJet30_v14',
    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFHT250_QuadPFJet25_PNet1BTag0p20_v7',
    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFHT250_QuadPFJet25_PNet2BTagMean0p55_v7',
    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFHT250_QuadPFJet25_v7',
    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFHT250_v7',
    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFHT280_QuadPFJet30_PNet2BTagMean0p55_v10',
    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFHT280_QuadPFJet30_v10',
    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFHT280_v10',
    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_v26',
    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_v24'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsBTagMuEGTau_datasetTau_selector
streamPhysicsBTagMuEGTau_datasetTau_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsBTagMuEGTau_datasetTau_selector.l1tResults = cms.InputTag('')
streamPhysicsBTagMuEGTau_datasetTau_selector.throw      = cms.bool(False)
streamPhysicsBTagMuEGTau_datasetTau_selector.triggerConditions = cms.vstring(
    'HLT_DoubleMediumChargedIsoDisplacedPFTauHPS32_Trk1_eta2p1_noDxy_v9',
    'HLT_DoubleMediumChargedIsoDisplacedPFTauHPS32_Trk1_eta2p1_v14',
    'HLT_DoubleMediumChargedIsoDisplacedPFTauHPS36_Trk1_eta2p1_v9',
    'HLT_DoublePNetTauhPFJet26_L2NN_eta2p3_PFJet60_v7',
    'HLT_DoublePNetTauhPFJet26_L2NN_eta2p3_PFJet75_v7',
    'HLT_DoublePNetTauhPFJet30_Medium_L2NN_eta2p3_v7',
    'HLT_DoublePNetTauhPFJet30_Tight_L2NN_eta2p3_v7',
    'HLT_SinglePNetTauhPFJet130_Loose_L2NN_eta2p3_v7',
    'HLT_SinglePNetTauhPFJet130_Medium_L2NN_eta2p3_v7',
    'HLT_SinglePNetTauhPFJet130_Tight_L2NN_eta2p3_v7'
)


# stream PhysicsCommissioning

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetCommissioning_selector
streamPhysicsCommissioning_datasetCommissioning_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetCommissioning_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetCommissioning_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetCommissioning_selector.triggerConditions = cms.vstring(
    'HLT_IsoTrackHB_v15',
    'HLT_IsoTrackHE_v15',
    'HLT_L1_CDC_SingleMu_3_er1p2_TOP120_DPHI2p618_3p142_v10',
    'HLT_PFJet40_GPUvsCPU_v8'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetCosmics_selector
streamPhysicsCommissioning_datasetCosmics_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetCosmics_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetCosmics_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetCosmics_selector.triggerConditions = cms.vstring('HLT_L1SingleMuCosmics_v9')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetHLTPhysics_selector
streamPhysicsCommissioning_datasetHLTPhysics_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetHLTPhysics_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetHLTPhysics_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetHLTPhysics_selector.triggerConditions = cms.vstring('HLT_Physics_v15')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetHcalNZS_selector
streamPhysicsCommissioning_datasetHcalNZS_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetHcalNZS_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetHcalNZS_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetHcalNZS_selector.triggerConditions = cms.vstring(
    'HLT_HcalNZS_v22',
    'HLT_HcalPhiSym_v24'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetMonteCarlo_selector
streamPhysicsCommissioning_datasetMonteCarlo_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetMonteCarlo_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetMonteCarlo_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetMonteCarlo_selector.triggerConditions = cms.vstring(
    'MC_AK4CaloJetsFromPV_v19',
    'MC_AK4CaloJets_v20',
    'MC_AK4PFJetPNet_v6',
    'MC_AK4PFJets_v30',
    'MC_AK8CaloHT_v19',
    'MC_AK8PFHT_v29',
    'MC_AK8PFJetPNet_v6',
    'MC_AK8PFJets_v30',
    'MC_CaloHT_v19',
    'MC_CaloMET_JetIdCleaned_v20',
    'MC_CaloMET_v19',
    'MC_CaloMHT_v19',
    'MC_Diphoton10_10_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass10_v24',
    'MC_DoubleEle5_CaloIdL_MW_v27',
    'MC_DoubleMuNoFiltersNoVtx_v18',
    'MC_DoubleMu_TrkIsoVVL_DZ_v24',
    'MC_Egamma_Open_Unseeded_v9',
    'MC_Egamma_Open_v9',
    'MC_Ele15_Ele10_CaloIdL_TrackIdL_IsoVL_DZ_v26',
    'MC_Ele5_WPTight_Gsf_v19',
    'MC_IsoMu_v28',
    'MC_PFHT_v29',
    'MC_PFMET_v30',
    'MC_PFMHT_v29',
    'MC_PFScouting_v7',
    'MC_ReducedIterativeTracking_v23'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetNoBPTX_selector
streamPhysicsCommissioning_datasetNoBPTX_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetNoBPTX_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetNoBPTX_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetNoBPTX_selector.triggerConditions = cms.vstring(
    'HLT_CDC_L2cosmic_10_er1p0_v11',
    'HLT_CDC_L2cosmic_5p5_er1p0_v11',
    'HLT_L2Mu10_NoVertex_NoBPTX3BX_v15',
    'HLT_L2Mu10_NoVertex_NoBPTX_v16',
    'HLT_L2Mu40_NoVertex_3Sta_NoBPTX3BX_v15',
    'HLT_L2Mu45_NoVertex_3Sta_NoBPTX3BX_v14',
    'HLT_UncorrectedJetE30_NoBPTX3BX_v15',
    'HLT_UncorrectedJetE30_NoBPTX_v15',
    'HLT_UncorrectedJetE60_NoBPTX3BX_v15',
    'HLT_UncorrectedJetE70_NoBPTX3BX_v15'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetZeroBias_selector
streamPhysicsCommissioning_datasetZeroBias_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetZeroBias_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetZeroBias_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetZeroBias_selector.triggerConditions = cms.vstring(
    'HLT_Random_v3',
    'HLT_ZeroBias_Alignment_v9',
    'HLT_ZeroBias_FirstBXAfterTrain_v11',
    'HLT_ZeroBias_FirstCollisionAfterAbortGap_v13',
    'HLT_ZeroBias_FirstCollisionInTrain_v12',
    'HLT_ZeroBias_IsolatedBunches_v13',
    'HLT_ZeroBias_LastCollisionInTrain_v11',
    'HLT_ZeroBias_v14'
)


# stream PhysicsEGamma0

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsEGamma0_datasetEGamma0_selector
streamPhysicsEGamma0_datasetEGamma0_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsEGamma0_datasetEGamma0_selector.l1tResults = cms.InputTag('')
streamPhysicsEGamma0_datasetEGamma0_selector.throw      = cms.bool(False)
streamPhysicsEGamma0_datasetEGamma0_selector.triggerConditions = cms.vstring(
    'HLT_DiEle27_WPTightCaloOnly_L1DoubleEG_v15',
    'HLT_DiPhoton10Time1ns_v11',
    'HLT_DiPhoton10Time1p2ns_v11',
    'HLT_DiPhoton10Time1p4ns_v11',
    'HLT_DiPhoton10Time1p6ns_v11',
    'HLT_DiPhoton10Time1p8ns_v11',
    'HLT_DiPhoton10Time2ns_v11',
    'HLT_DiPhoton10_CaloIdL_v11',
    'HLT_DiSC30_18_EIso_AND_HE_Mass70_v25',
    'HLT_Diphoton20_14_eta1p5_R9IdL_AND_HET_AND_IsoTCaloIdT_v11',
    'HLT_Diphoton20_14_eta1p5_R9IdL_AND_HE_AND_IsoTCaloIdT_v11',
    'HLT_Diphoton22_14_eta1p5_R9IdL_AND_HET_AND_IsoTCaloIdT_v11',
    'HLT_Diphoton22_14_eta1p5_R9IdL_AND_HE_AND_IsoTCaloIdT_v11',
    'HLT_Diphoton24_14_eta1p5_R9IdL_AND_HET_AND_IsoTCaloIdT_v11',
    'HLT_Diphoton24_16_eta1p5_R9IdL_AND_HET_AND_IsoTCaloIdT_v11',
    'HLT_Diphoton30_18_R9IdL_AND_HE_AND_IsoCaloId_Mass55_v12',
    'HLT_Diphoton30_18_R9IdL_AND_HE_AND_IsoCaloId_v12',
    'HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass90_v24',
    'HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass95_v24',
    'HLT_DiphotonMVA14p25_High_Mass60_v1',
    'HLT_DiphotonMVA14p25_Low_Mass60_v1',
    'HLT_DiphotonMVA14p25_Medium_Mass60_v1',
    'HLT_DiphotonMVA14p25_TightHigh_Mass60_v1',
    'HLT_DiphotonMVA14p25_TightLow_Mass60_v1',
    'HLT_DiphotonMVA14p25_TightMedium_Mass60_v1',
    'HLT_DoubleEle10_eta1p22_mMax6_v11',
    'HLT_DoubleEle25_CaloIdL_MW_v16',
    'HLT_DoubleEle27_CaloIdL_MW_v16',
    'HLT_DoubleEle33_CaloIdL_MW_v29',
    'HLT_DoubleEle6p5_eta1p22_mMax6_v11',
    'HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_DZ_PFHT350_v33',
    'HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_PFHT350_v33',
    'HLT_DoubleEle8_eta1p22_mMax6_v11',
    'HLT_DoublePhoton33_CaloIdL_v18',
    'HLT_DoublePhoton70_v18',
    'HLT_DoublePhoton85_v26',
    'HLT_ECALHT800_v21',
    'HLT_Ele115_CaloIdVT_GsfTrkIdT_v26',
    'HLT_Ele12_CaloIdL_TrackIdL_IsoVL_PFJet30_v31',
    'HLT_Ele135_CaloIdVT_GsfTrkIdT_v19',
    'HLT_Ele14_eta2p5_IsoVVVL_Gsf_PFHT200_PNetBTag0p53_v6',
    'HLT_Ele15_IsoVVVL_PFHT450_PFMET50_v29',
    'HLT_Ele15_IsoVVVL_PFHT450_v29',
    'HLT_Ele15_IsoVVVL_PFHT600_v33',
    'HLT_Ele17_CaloIdM_TrackIdM_PFJet30_v29',
    'HLT_Ele23_CaloIdL_TrackIdL_IsoVL_PFJet30_v31',
    'HLT_Ele23_CaloIdM_TrackIdM_PFJet30_v31',
    'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v30',
    'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_v30',
    'HLT_Ele24_eta2p1_WPTight_Gsf_PNetTauhPFJet30_Loose_eta2p3_CrossL1_v7',
    'HLT_Ele24_eta2p1_WPTight_Gsf_PNetTauhPFJet30_Medium_eta2p3_CrossL1_v7',
    'HLT_Ele24_eta2p1_WPTight_Gsf_PNetTauhPFJet30_Tight_eta2p3_CrossL1_v7',
    'HLT_Ele28_HighEta_SC20_Mass55_v24',
    'HLT_Ele30_WPTight_Gsf_v12',
    'HLT_Ele32_WPTight_Gsf_L1DoubleEG_v20',
    'HLT_Ele32_WPTight_Gsf_v26',
    'HLT_Ele35_WPTight_Gsf_v20',
    'HLT_Ele38_WPTight_Gsf_v20',
    'HLT_Ele40_WPTight_Gsf_v20',
    'HLT_Ele50_CaloIdVT_GsfTrkIdT_AK8PFJet220_SoftDropMass40_PNetBB0p06_v10',
    'HLT_Ele50_CaloIdVT_GsfTrkIdT_AK8PFJet220_SoftDropMass40_v13',
    'HLT_Ele50_CaloIdVT_GsfTrkIdT_AK8PFJet230_SoftDropMass40_PNetBB0p06_v10',
    'HLT_Ele50_CaloIdVT_GsfTrkIdT_AK8PFJet230_SoftDropMass40_PNetBB0p10_v10',
    'HLT_Ele50_CaloIdVT_GsfTrkIdT_AK8PFJet230_SoftDropMass40_v13',
    'HLT_Ele50_CaloIdVT_GsfTrkIdT_PFJet165_v31',
    'HLT_Ele50_IsoVVVL_PFHT450_v29',
    'HLT_Ele8_CaloIdL_TrackIdL_IsoVL_PFJet30_v29',
    'HLT_Ele8_CaloIdM_TrackIdM_PFJet30_v31',
    'HLT_Photon100EBHE10_v13',
    'HLT_Photon110EB_TightID_TightIso_AK8CaloJet30_v4',
    'HLT_Photon110EB_TightID_TightIso_AK8PFJet30_v6',
    'HLT_Photon110EB_TightID_TightIso_CaloJet30_v4',
    'HLT_Photon110EB_TightID_TightIso_PFJet30_v7',
    'HLT_Photon110EB_TightID_TightIso_v13',
    'HLT_Photon120_R9Id90_HE10_IsoM_v25',
    'HLT_Photon120_v24',
    'HLT_Photon150_v18',
    'HLT_Photon165_R9Id90_HE10_IsoM_v26',
    'HLT_Photon175_v26',
    'HLT_Photon200_v25',
    'HLT_Photon20_HoverELoose_v21',
    'HLT_Photon300_NoHE_v24',
    'HLT_Photon30EB_TightID_TightIso_v13',
    'HLT_Photon30_HoverELoose_v21',
    'HLT_Photon32_OneProng32_M50To105_v11',
    'HLT_Photon33_v16',
    'HLT_Photon34_R9Id90_CaloIdL_IsoL_DisplacedIdL_MediumChargedIsoDisplacedPFTauHPS34_v9',
    'HLT_Photon35_TwoProngs35_v14',
    'HLT_Photon40EB_TightID_TightIso_v4',
    'HLT_Photon40EB_v4',
    'HLT_Photon45EB_TightID_TightIso_v4',
    'HLT_Photon45EB_v4',
    'HLT_Photon50EB_TightID_TightIso_AK8CaloJet30_v4',
    'HLT_Photon50EB_TightID_TightIso_AK8PFJet30_v6',
    'HLT_Photon50EB_TightID_TightIso_CaloJet30_v4',
    'HLT_Photon50EB_TightID_TightIso_PFJet30_v7',
    'HLT_Photon50EB_TightID_TightIso_v9',
    'HLT_Photon50EB_v5',
    'HLT_Photon50_R9Id90_HE10_IsoM_v25',
    'HLT_Photon50_TimeGt2p5ns_v8',
    'HLT_Photon50_TimeLtNeg2p5ns_v8',
    'HLT_Photon50_v24',
    'HLT_Photon55EB_TightID_TightIso_v5',
    'HLT_Photon60_R9Id90_CaloIdL_IsoL_DisplacedIdL_PFHT350_v11',
    'HLT_Photon60_R9Id90_CaloIdL_IsoL_DisplacedIdL_PFHT380_v11',
    'HLT_Photon60_R9Id90_CaloIdL_IsoL_DisplacedIdL_PFHT400_v11',
    'HLT_Photon60_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ300DEta3_v10',
    'HLT_Photon75EB_TightID_TightIso_v9',
    'HLT_Photon75_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ300DEta3_v18',
    'HLT_Photon75_R9Id90_HE10_IsoM_v25',
    'HLT_Photon75_v24',
    'HLT_Photon90EB_TightID_TightIso_v9',
    'HLT_Photon90_R9Id90_HE10_IsoM_v25',
    'HLT_Photon90_v24',
    'HLT_SingleEle8_SingleEGL1_v10',
    'HLT_SingleEle8_v10'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsEGamma0_datasetEGamma1_selector
streamPhysicsEGamma0_datasetEGamma1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsEGamma0_datasetEGamma1_selector.l1tResults = cms.InputTag('')
streamPhysicsEGamma0_datasetEGamma1_selector.throw      = cms.bool(False)
streamPhysicsEGamma0_datasetEGamma1_selector.triggerConditions = cms.vstring(
    'HLT_DiEle27_WPTightCaloOnly_L1DoubleEG_v15',
    'HLT_DiPhoton10Time1ns_v11',
    'HLT_DiPhoton10Time1p2ns_v11',
    'HLT_DiPhoton10Time1p4ns_v11',
    'HLT_DiPhoton10Time1p6ns_v11',
    'HLT_DiPhoton10Time1p8ns_v11',
    'HLT_DiPhoton10Time2ns_v11',
    'HLT_DiPhoton10_CaloIdL_v11',
    'HLT_DiSC30_18_EIso_AND_HE_Mass70_v25',
    'HLT_Diphoton20_14_eta1p5_R9IdL_AND_HET_AND_IsoTCaloIdT_v11',
    'HLT_Diphoton20_14_eta1p5_R9IdL_AND_HE_AND_IsoTCaloIdT_v11',
    'HLT_Diphoton22_14_eta1p5_R9IdL_AND_HET_AND_IsoTCaloIdT_v11',
    'HLT_Diphoton22_14_eta1p5_R9IdL_AND_HE_AND_IsoTCaloIdT_v11',
    'HLT_Diphoton24_14_eta1p5_R9IdL_AND_HET_AND_IsoTCaloIdT_v11',
    'HLT_Diphoton24_16_eta1p5_R9IdL_AND_HET_AND_IsoTCaloIdT_v11',
    'HLT_Diphoton30_18_R9IdL_AND_HE_AND_IsoCaloId_Mass55_v12',
    'HLT_Diphoton30_18_R9IdL_AND_HE_AND_IsoCaloId_v12',
    'HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass90_v24',
    'HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass95_v24',
    'HLT_DiphotonMVA14p25_High_Mass60_v1',
    'HLT_DiphotonMVA14p25_Low_Mass60_v1',
    'HLT_DiphotonMVA14p25_Medium_Mass60_v1',
    'HLT_DiphotonMVA14p25_TightHigh_Mass60_v1',
    'HLT_DiphotonMVA14p25_TightLow_Mass60_v1',
    'HLT_DiphotonMVA14p25_TightMedium_Mass60_v1',
    'HLT_DoubleEle10_eta1p22_mMax6_v11',
    'HLT_DoubleEle25_CaloIdL_MW_v16',
    'HLT_DoubleEle27_CaloIdL_MW_v16',
    'HLT_DoubleEle33_CaloIdL_MW_v29',
    'HLT_DoubleEle6p5_eta1p22_mMax6_v11',
    'HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_DZ_PFHT350_v33',
    'HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_PFHT350_v33',
    'HLT_DoubleEle8_eta1p22_mMax6_v11',
    'HLT_DoublePhoton33_CaloIdL_v18',
    'HLT_DoublePhoton70_v18',
    'HLT_DoublePhoton85_v26',
    'HLT_ECALHT800_v21',
    'HLT_Ele115_CaloIdVT_GsfTrkIdT_v26',
    'HLT_Ele12_CaloIdL_TrackIdL_IsoVL_PFJet30_v31',
    'HLT_Ele135_CaloIdVT_GsfTrkIdT_v19',
    'HLT_Ele14_eta2p5_IsoVVVL_Gsf_PFHT200_PNetBTag0p53_v6',
    'HLT_Ele15_IsoVVVL_PFHT450_PFMET50_v29',
    'HLT_Ele15_IsoVVVL_PFHT450_v29',
    'HLT_Ele15_IsoVVVL_PFHT600_v33',
    'HLT_Ele17_CaloIdM_TrackIdM_PFJet30_v29',
    'HLT_Ele23_CaloIdL_TrackIdL_IsoVL_PFJet30_v31',
    'HLT_Ele23_CaloIdM_TrackIdM_PFJet30_v31',
    'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v30',
    'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_v30',
    'HLT_Ele24_eta2p1_WPTight_Gsf_PNetTauhPFJet30_Loose_eta2p3_CrossL1_v7',
    'HLT_Ele24_eta2p1_WPTight_Gsf_PNetTauhPFJet30_Medium_eta2p3_CrossL1_v7',
    'HLT_Ele24_eta2p1_WPTight_Gsf_PNetTauhPFJet30_Tight_eta2p3_CrossL1_v7',
    'HLT_Ele28_HighEta_SC20_Mass55_v24',
    'HLT_Ele30_WPTight_Gsf_v12',
    'HLT_Ele32_WPTight_Gsf_L1DoubleEG_v20',
    'HLT_Ele32_WPTight_Gsf_v26',
    'HLT_Ele35_WPTight_Gsf_v20',
    'HLT_Ele38_WPTight_Gsf_v20',
    'HLT_Ele40_WPTight_Gsf_v20',
    'HLT_Ele50_CaloIdVT_GsfTrkIdT_AK8PFJet220_SoftDropMass40_PNetBB0p06_v10',
    'HLT_Ele50_CaloIdVT_GsfTrkIdT_AK8PFJet220_SoftDropMass40_v13',
    'HLT_Ele50_CaloIdVT_GsfTrkIdT_AK8PFJet230_SoftDropMass40_PNetBB0p06_v10',
    'HLT_Ele50_CaloIdVT_GsfTrkIdT_AK8PFJet230_SoftDropMass40_PNetBB0p10_v10',
    'HLT_Ele50_CaloIdVT_GsfTrkIdT_AK8PFJet230_SoftDropMass40_v13',
    'HLT_Ele50_CaloIdVT_GsfTrkIdT_PFJet165_v31',
    'HLT_Ele50_IsoVVVL_PFHT450_v29',
    'HLT_Ele8_CaloIdL_TrackIdL_IsoVL_PFJet30_v29',
    'HLT_Ele8_CaloIdM_TrackIdM_PFJet30_v31',
    'HLT_Photon100EBHE10_v13',
    'HLT_Photon110EB_TightID_TightIso_AK8CaloJet30_v4',
    'HLT_Photon110EB_TightID_TightIso_AK8PFJet30_v6',
    'HLT_Photon110EB_TightID_TightIso_CaloJet30_v4',
    'HLT_Photon110EB_TightID_TightIso_PFJet30_v7',
    'HLT_Photon110EB_TightID_TightIso_v13',
    'HLT_Photon120_R9Id90_HE10_IsoM_v25',
    'HLT_Photon120_v24',
    'HLT_Photon150_v18',
    'HLT_Photon165_R9Id90_HE10_IsoM_v26',
    'HLT_Photon175_v26',
    'HLT_Photon200_v25',
    'HLT_Photon20_HoverELoose_v21',
    'HLT_Photon300_NoHE_v24',
    'HLT_Photon30EB_TightID_TightIso_v13',
    'HLT_Photon30_HoverELoose_v21',
    'HLT_Photon32_OneProng32_M50To105_v11',
    'HLT_Photon33_v16',
    'HLT_Photon34_R9Id90_CaloIdL_IsoL_DisplacedIdL_MediumChargedIsoDisplacedPFTauHPS34_v9',
    'HLT_Photon35_TwoProngs35_v14',
    'HLT_Photon40EB_TightID_TightIso_v4',
    'HLT_Photon40EB_v4',
    'HLT_Photon45EB_TightID_TightIso_v4',
    'HLT_Photon45EB_v4',
    'HLT_Photon50EB_TightID_TightIso_AK8CaloJet30_v4',
    'HLT_Photon50EB_TightID_TightIso_AK8PFJet30_v6',
    'HLT_Photon50EB_TightID_TightIso_CaloJet30_v4',
    'HLT_Photon50EB_TightID_TightIso_PFJet30_v7',
    'HLT_Photon50EB_TightID_TightIso_v9',
    'HLT_Photon50EB_v5',
    'HLT_Photon50_R9Id90_HE10_IsoM_v25',
    'HLT_Photon50_TimeGt2p5ns_v8',
    'HLT_Photon50_TimeLtNeg2p5ns_v8',
    'HLT_Photon50_v24',
    'HLT_Photon55EB_TightID_TightIso_v5',
    'HLT_Photon60_R9Id90_CaloIdL_IsoL_DisplacedIdL_PFHT350_v11',
    'HLT_Photon60_R9Id90_CaloIdL_IsoL_DisplacedIdL_PFHT380_v11',
    'HLT_Photon60_R9Id90_CaloIdL_IsoL_DisplacedIdL_PFHT400_v11',
    'HLT_Photon60_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ300DEta3_v10',
    'HLT_Photon75EB_TightID_TightIso_v9',
    'HLT_Photon75_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ300DEta3_v18',
    'HLT_Photon75_R9Id90_HE10_IsoM_v25',
    'HLT_Photon75_v24',
    'HLT_Photon90EB_TightID_TightIso_v9',
    'HLT_Photon90_R9Id90_HE10_IsoM_v25',
    'HLT_Photon90_v24',
    'HLT_SingleEle8_SingleEGL1_v10',
    'HLT_SingleEle8_v10'
)


# stream PhysicsEGamma1

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsEGamma1_datasetEGamma2_selector
streamPhysicsEGamma1_datasetEGamma2_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsEGamma1_datasetEGamma2_selector.l1tResults = cms.InputTag('')
streamPhysicsEGamma1_datasetEGamma2_selector.throw      = cms.bool(False)
streamPhysicsEGamma1_datasetEGamma2_selector.triggerConditions = cms.vstring(
    'HLT_DiEle27_WPTightCaloOnly_L1DoubleEG_v15',
    'HLT_DiPhoton10Time1ns_v11',
    'HLT_DiPhoton10Time1p2ns_v11',
    'HLT_DiPhoton10Time1p4ns_v11',
    'HLT_DiPhoton10Time1p6ns_v11',
    'HLT_DiPhoton10Time1p8ns_v11',
    'HLT_DiPhoton10Time2ns_v11',
    'HLT_DiPhoton10_CaloIdL_v11',
    'HLT_DiSC30_18_EIso_AND_HE_Mass70_v25',
    'HLT_Diphoton20_14_eta1p5_R9IdL_AND_HET_AND_IsoTCaloIdT_v11',
    'HLT_Diphoton20_14_eta1p5_R9IdL_AND_HE_AND_IsoTCaloIdT_v11',
    'HLT_Diphoton22_14_eta1p5_R9IdL_AND_HET_AND_IsoTCaloIdT_v11',
    'HLT_Diphoton22_14_eta1p5_R9IdL_AND_HE_AND_IsoTCaloIdT_v11',
    'HLT_Diphoton24_14_eta1p5_R9IdL_AND_HET_AND_IsoTCaloIdT_v11',
    'HLT_Diphoton24_16_eta1p5_R9IdL_AND_HET_AND_IsoTCaloIdT_v11',
    'HLT_Diphoton30_18_R9IdL_AND_HE_AND_IsoCaloId_Mass55_v12',
    'HLT_Diphoton30_18_R9IdL_AND_HE_AND_IsoCaloId_v12',
    'HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass90_v24',
    'HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass95_v24',
    'HLT_DiphotonMVA14p25_High_Mass60_v1',
    'HLT_DiphotonMVA14p25_Low_Mass60_v1',
    'HLT_DiphotonMVA14p25_Medium_Mass60_v1',
    'HLT_DiphotonMVA14p25_TightHigh_Mass60_v1',
    'HLT_DiphotonMVA14p25_TightLow_Mass60_v1',
    'HLT_DiphotonMVA14p25_TightMedium_Mass60_v1',
    'HLT_DoubleEle10_eta1p22_mMax6_v11',
    'HLT_DoubleEle25_CaloIdL_MW_v16',
    'HLT_DoubleEle27_CaloIdL_MW_v16',
    'HLT_DoubleEle33_CaloIdL_MW_v29',
    'HLT_DoubleEle6p5_eta1p22_mMax6_v11',
    'HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_DZ_PFHT350_v33',
    'HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_PFHT350_v33',
    'HLT_DoubleEle8_eta1p22_mMax6_v11',
    'HLT_DoublePhoton33_CaloIdL_v18',
    'HLT_DoublePhoton70_v18',
    'HLT_DoublePhoton85_v26',
    'HLT_ECALHT800_v21',
    'HLT_Ele115_CaloIdVT_GsfTrkIdT_v26',
    'HLT_Ele12_CaloIdL_TrackIdL_IsoVL_PFJet30_v31',
    'HLT_Ele135_CaloIdVT_GsfTrkIdT_v19',
    'HLT_Ele14_eta2p5_IsoVVVL_Gsf_PFHT200_PNetBTag0p53_v6',
    'HLT_Ele15_IsoVVVL_PFHT450_PFMET50_v29',
    'HLT_Ele15_IsoVVVL_PFHT450_v29',
    'HLT_Ele15_IsoVVVL_PFHT600_v33',
    'HLT_Ele17_CaloIdM_TrackIdM_PFJet30_v29',
    'HLT_Ele23_CaloIdL_TrackIdL_IsoVL_PFJet30_v31',
    'HLT_Ele23_CaloIdM_TrackIdM_PFJet30_v31',
    'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v30',
    'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_v30',
    'HLT_Ele24_eta2p1_WPTight_Gsf_PNetTauhPFJet30_Loose_eta2p3_CrossL1_v7',
    'HLT_Ele24_eta2p1_WPTight_Gsf_PNetTauhPFJet30_Medium_eta2p3_CrossL1_v7',
    'HLT_Ele24_eta2p1_WPTight_Gsf_PNetTauhPFJet30_Tight_eta2p3_CrossL1_v7',
    'HLT_Ele28_HighEta_SC20_Mass55_v24',
    'HLT_Ele30_WPTight_Gsf_v12',
    'HLT_Ele32_WPTight_Gsf_L1DoubleEG_v20',
    'HLT_Ele32_WPTight_Gsf_v26',
    'HLT_Ele35_WPTight_Gsf_v20',
    'HLT_Ele38_WPTight_Gsf_v20',
    'HLT_Ele40_WPTight_Gsf_v20',
    'HLT_Ele50_CaloIdVT_GsfTrkIdT_AK8PFJet220_SoftDropMass40_PNetBB0p06_v10',
    'HLT_Ele50_CaloIdVT_GsfTrkIdT_AK8PFJet220_SoftDropMass40_v13',
    'HLT_Ele50_CaloIdVT_GsfTrkIdT_AK8PFJet230_SoftDropMass40_PNetBB0p06_v10',
    'HLT_Ele50_CaloIdVT_GsfTrkIdT_AK8PFJet230_SoftDropMass40_PNetBB0p10_v10',
    'HLT_Ele50_CaloIdVT_GsfTrkIdT_AK8PFJet230_SoftDropMass40_v13',
    'HLT_Ele50_CaloIdVT_GsfTrkIdT_PFJet165_v31',
    'HLT_Ele50_IsoVVVL_PFHT450_v29',
    'HLT_Ele8_CaloIdL_TrackIdL_IsoVL_PFJet30_v29',
    'HLT_Ele8_CaloIdM_TrackIdM_PFJet30_v31',
    'HLT_Photon100EBHE10_v13',
    'HLT_Photon110EB_TightID_TightIso_AK8CaloJet30_v4',
    'HLT_Photon110EB_TightID_TightIso_AK8PFJet30_v6',
    'HLT_Photon110EB_TightID_TightIso_CaloJet30_v4',
    'HLT_Photon110EB_TightID_TightIso_PFJet30_v7',
    'HLT_Photon110EB_TightID_TightIso_v13',
    'HLT_Photon120_R9Id90_HE10_IsoM_v25',
    'HLT_Photon120_v24',
    'HLT_Photon150_v18',
    'HLT_Photon165_R9Id90_HE10_IsoM_v26',
    'HLT_Photon175_v26',
    'HLT_Photon200_v25',
    'HLT_Photon20_HoverELoose_v21',
    'HLT_Photon300_NoHE_v24',
    'HLT_Photon30EB_TightID_TightIso_v13',
    'HLT_Photon30_HoverELoose_v21',
    'HLT_Photon32_OneProng32_M50To105_v11',
    'HLT_Photon33_v16',
    'HLT_Photon34_R9Id90_CaloIdL_IsoL_DisplacedIdL_MediumChargedIsoDisplacedPFTauHPS34_v9',
    'HLT_Photon35_TwoProngs35_v14',
    'HLT_Photon40EB_TightID_TightIso_v4',
    'HLT_Photon40EB_v4',
    'HLT_Photon45EB_TightID_TightIso_v4',
    'HLT_Photon45EB_v4',
    'HLT_Photon50EB_TightID_TightIso_AK8CaloJet30_v4',
    'HLT_Photon50EB_TightID_TightIso_AK8PFJet30_v6',
    'HLT_Photon50EB_TightID_TightIso_CaloJet30_v4',
    'HLT_Photon50EB_TightID_TightIso_PFJet30_v7',
    'HLT_Photon50EB_TightID_TightIso_v9',
    'HLT_Photon50EB_v5',
    'HLT_Photon50_R9Id90_HE10_IsoM_v25',
    'HLT_Photon50_TimeGt2p5ns_v8',
    'HLT_Photon50_TimeLtNeg2p5ns_v8',
    'HLT_Photon50_v24',
    'HLT_Photon55EB_TightID_TightIso_v5',
    'HLT_Photon60_R9Id90_CaloIdL_IsoL_DisplacedIdL_PFHT350_v11',
    'HLT_Photon60_R9Id90_CaloIdL_IsoL_DisplacedIdL_PFHT380_v11',
    'HLT_Photon60_R9Id90_CaloIdL_IsoL_DisplacedIdL_PFHT400_v11',
    'HLT_Photon60_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ300DEta3_v10',
    'HLT_Photon75EB_TightID_TightIso_v9',
    'HLT_Photon75_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ300DEta3_v18',
    'HLT_Photon75_R9Id90_HE10_IsoM_v25',
    'HLT_Photon75_v24',
    'HLT_Photon90EB_TightID_TightIso_v9',
    'HLT_Photon90_R9Id90_HE10_IsoM_v25',
    'HLT_Photon90_v24',
    'HLT_SingleEle8_SingleEGL1_v10',
    'HLT_SingleEle8_v10'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsEGamma1_datasetEGamma3_selector
streamPhysicsEGamma1_datasetEGamma3_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsEGamma1_datasetEGamma3_selector.l1tResults = cms.InputTag('')
streamPhysicsEGamma1_datasetEGamma3_selector.throw      = cms.bool(False)
streamPhysicsEGamma1_datasetEGamma3_selector.triggerConditions = cms.vstring(
    'HLT_DiEle27_WPTightCaloOnly_L1DoubleEG_v15',
    'HLT_DiPhoton10Time1ns_v11',
    'HLT_DiPhoton10Time1p2ns_v11',
    'HLT_DiPhoton10Time1p4ns_v11',
    'HLT_DiPhoton10Time1p6ns_v11',
    'HLT_DiPhoton10Time1p8ns_v11',
    'HLT_DiPhoton10Time2ns_v11',
    'HLT_DiPhoton10_CaloIdL_v11',
    'HLT_DiSC30_18_EIso_AND_HE_Mass70_v25',
    'HLT_Diphoton20_14_eta1p5_R9IdL_AND_HET_AND_IsoTCaloIdT_v11',
    'HLT_Diphoton20_14_eta1p5_R9IdL_AND_HE_AND_IsoTCaloIdT_v11',
    'HLT_Diphoton22_14_eta1p5_R9IdL_AND_HET_AND_IsoTCaloIdT_v11',
    'HLT_Diphoton22_14_eta1p5_R9IdL_AND_HE_AND_IsoTCaloIdT_v11',
    'HLT_Diphoton24_14_eta1p5_R9IdL_AND_HET_AND_IsoTCaloIdT_v11',
    'HLT_Diphoton24_16_eta1p5_R9IdL_AND_HET_AND_IsoTCaloIdT_v11',
    'HLT_Diphoton30_18_R9IdL_AND_HE_AND_IsoCaloId_Mass55_v12',
    'HLT_Diphoton30_18_R9IdL_AND_HE_AND_IsoCaloId_v12',
    'HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass90_v24',
    'HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass95_v24',
    'HLT_DiphotonMVA14p25_High_Mass60_v1',
    'HLT_DiphotonMVA14p25_Low_Mass60_v1',
    'HLT_DiphotonMVA14p25_Medium_Mass60_v1',
    'HLT_DiphotonMVA14p25_TightHigh_Mass60_v1',
    'HLT_DiphotonMVA14p25_TightLow_Mass60_v1',
    'HLT_DiphotonMVA14p25_TightMedium_Mass60_v1',
    'HLT_DoubleEle10_eta1p22_mMax6_v11',
    'HLT_DoubleEle25_CaloIdL_MW_v16',
    'HLT_DoubleEle27_CaloIdL_MW_v16',
    'HLT_DoubleEle33_CaloIdL_MW_v29',
    'HLT_DoubleEle6p5_eta1p22_mMax6_v11',
    'HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_DZ_PFHT350_v33',
    'HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_PFHT350_v33',
    'HLT_DoubleEle8_eta1p22_mMax6_v11',
    'HLT_DoublePhoton33_CaloIdL_v18',
    'HLT_DoublePhoton70_v18',
    'HLT_DoublePhoton85_v26',
    'HLT_ECALHT800_v21',
    'HLT_Ele115_CaloIdVT_GsfTrkIdT_v26',
    'HLT_Ele12_CaloIdL_TrackIdL_IsoVL_PFJet30_v31',
    'HLT_Ele135_CaloIdVT_GsfTrkIdT_v19',
    'HLT_Ele14_eta2p5_IsoVVVL_Gsf_PFHT200_PNetBTag0p53_v6',
    'HLT_Ele15_IsoVVVL_PFHT450_PFMET50_v29',
    'HLT_Ele15_IsoVVVL_PFHT450_v29',
    'HLT_Ele15_IsoVVVL_PFHT600_v33',
    'HLT_Ele17_CaloIdM_TrackIdM_PFJet30_v29',
    'HLT_Ele23_CaloIdL_TrackIdL_IsoVL_PFJet30_v31',
    'HLT_Ele23_CaloIdM_TrackIdM_PFJet30_v31',
    'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v30',
    'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_v30',
    'HLT_Ele24_eta2p1_WPTight_Gsf_PNetTauhPFJet30_Loose_eta2p3_CrossL1_v7',
    'HLT_Ele24_eta2p1_WPTight_Gsf_PNetTauhPFJet30_Medium_eta2p3_CrossL1_v7',
    'HLT_Ele24_eta2p1_WPTight_Gsf_PNetTauhPFJet30_Tight_eta2p3_CrossL1_v7',
    'HLT_Ele28_HighEta_SC20_Mass55_v24',
    'HLT_Ele30_WPTight_Gsf_v12',
    'HLT_Ele32_WPTight_Gsf_L1DoubleEG_v20',
    'HLT_Ele32_WPTight_Gsf_v26',
    'HLT_Ele35_WPTight_Gsf_v20',
    'HLT_Ele38_WPTight_Gsf_v20',
    'HLT_Ele40_WPTight_Gsf_v20',
    'HLT_Ele50_CaloIdVT_GsfTrkIdT_AK8PFJet220_SoftDropMass40_PNetBB0p06_v10',
    'HLT_Ele50_CaloIdVT_GsfTrkIdT_AK8PFJet220_SoftDropMass40_v13',
    'HLT_Ele50_CaloIdVT_GsfTrkIdT_AK8PFJet230_SoftDropMass40_PNetBB0p06_v10',
    'HLT_Ele50_CaloIdVT_GsfTrkIdT_AK8PFJet230_SoftDropMass40_PNetBB0p10_v10',
    'HLT_Ele50_CaloIdVT_GsfTrkIdT_AK8PFJet230_SoftDropMass40_v13',
    'HLT_Ele50_CaloIdVT_GsfTrkIdT_PFJet165_v31',
    'HLT_Ele50_IsoVVVL_PFHT450_v29',
    'HLT_Ele8_CaloIdL_TrackIdL_IsoVL_PFJet30_v29',
    'HLT_Ele8_CaloIdM_TrackIdM_PFJet30_v31',
    'HLT_Photon100EBHE10_v13',
    'HLT_Photon110EB_TightID_TightIso_AK8CaloJet30_v4',
    'HLT_Photon110EB_TightID_TightIso_AK8PFJet30_v6',
    'HLT_Photon110EB_TightID_TightIso_CaloJet30_v4',
    'HLT_Photon110EB_TightID_TightIso_PFJet30_v7',
    'HLT_Photon110EB_TightID_TightIso_v13',
    'HLT_Photon120_R9Id90_HE10_IsoM_v25',
    'HLT_Photon120_v24',
    'HLT_Photon150_v18',
    'HLT_Photon165_R9Id90_HE10_IsoM_v26',
    'HLT_Photon175_v26',
    'HLT_Photon200_v25',
    'HLT_Photon20_HoverELoose_v21',
    'HLT_Photon300_NoHE_v24',
    'HLT_Photon30EB_TightID_TightIso_v13',
    'HLT_Photon30_HoverELoose_v21',
    'HLT_Photon32_OneProng32_M50To105_v11',
    'HLT_Photon33_v16',
    'HLT_Photon34_R9Id90_CaloIdL_IsoL_DisplacedIdL_MediumChargedIsoDisplacedPFTauHPS34_v9',
    'HLT_Photon35_TwoProngs35_v14',
    'HLT_Photon40EB_TightID_TightIso_v4',
    'HLT_Photon40EB_v4',
    'HLT_Photon45EB_TightID_TightIso_v4',
    'HLT_Photon45EB_v4',
    'HLT_Photon50EB_TightID_TightIso_AK8CaloJet30_v4',
    'HLT_Photon50EB_TightID_TightIso_AK8PFJet30_v6',
    'HLT_Photon50EB_TightID_TightIso_CaloJet30_v4',
    'HLT_Photon50EB_TightID_TightIso_PFJet30_v7',
    'HLT_Photon50EB_TightID_TightIso_v9',
    'HLT_Photon50EB_v5',
    'HLT_Photon50_R9Id90_HE10_IsoM_v25',
    'HLT_Photon50_TimeGt2p5ns_v8',
    'HLT_Photon50_TimeLtNeg2p5ns_v8',
    'HLT_Photon50_v24',
    'HLT_Photon55EB_TightID_TightIso_v5',
    'HLT_Photon60_R9Id90_CaloIdL_IsoL_DisplacedIdL_PFHT350_v11',
    'HLT_Photon60_R9Id90_CaloIdL_IsoL_DisplacedIdL_PFHT380_v11',
    'HLT_Photon60_R9Id90_CaloIdL_IsoL_DisplacedIdL_PFHT400_v11',
    'HLT_Photon60_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ300DEta3_v10',
    'HLT_Photon75EB_TightID_TightIso_v9',
    'HLT_Photon75_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ300DEta3_v18',
    'HLT_Photon75_R9Id90_HE10_IsoM_v25',
    'HLT_Photon75_v24',
    'HLT_Photon90EB_TightID_TightIso_v9',
    'HLT_Photon90_R9Id90_HE10_IsoM_v25',
    'HLT_Photon90_v24',
    'HLT_SingleEle8_SingleEGL1_v10',
    'HLT_SingleEle8_v10'
)


# stream PhysicsEmittanceScan0

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsEmittanceScan0_datasetEmittanceScan0_selector
streamPhysicsEmittanceScan0_datasetEmittanceScan0_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsEmittanceScan0_datasetEmittanceScan0_selector.l1tResults = cms.InputTag('')
streamPhysicsEmittanceScan0_datasetEmittanceScan0_selector.throw      = cms.bool(False)
streamPhysicsEmittanceScan0_datasetEmittanceScan0_selector.triggerConditions = cms.vstring('HLT_L1AlwaysTrue_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsEmittanceScan0_datasetEmittanceScan1_selector
streamPhysicsEmittanceScan0_datasetEmittanceScan1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsEmittanceScan0_datasetEmittanceScan1_selector.l1tResults = cms.InputTag('')
streamPhysicsEmittanceScan0_datasetEmittanceScan1_selector.throw      = cms.bool(False)
streamPhysicsEmittanceScan0_datasetEmittanceScan1_selector.triggerConditions = cms.vstring('HLT_L1AlwaysTrue_v1')


# stream PhysicsEmittanceScan1

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsEmittanceScan1_datasetEmittanceScan2_selector
streamPhysicsEmittanceScan1_datasetEmittanceScan2_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsEmittanceScan1_datasetEmittanceScan2_selector.l1tResults = cms.InputTag('')
streamPhysicsEmittanceScan1_datasetEmittanceScan2_selector.throw      = cms.bool(False)
streamPhysicsEmittanceScan1_datasetEmittanceScan2_selector.triggerConditions = cms.vstring('HLT_L1AlwaysTrue_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsEmittanceScan1_datasetEmittanceScan3_selector
streamPhysicsEmittanceScan1_datasetEmittanceScan3_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsEmittanceScan1_datasetEmittanceScan3_selector.l1tResults = cms.InputTag('')
streamPhysicsEmittanceScan1_datasetEmittanceScan3_selector.throw      = cms.bool(False)
streamPhysicsEmittanceScan1_datasetEmittanceScan3_selector.triggerConditions = cms.vstring('HLT_L1AlwaysTrue_v1')


# stream PhysicsEmittanceScan2

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsEmittanceScan2_datasetEmittanceScan4_selector
streamPhysicsEmittanceScan2_datasetEmittanceScan4_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsEmittanceScan2_datasetEmittanceScan4_selector.l1tResults = cms.InputTag('')
streamPhysicsEmittanceScan2_datasetEmittanceScan4_selector.throw      = cms.bool(False)
streamPhysicsEmittanceScan2_datasetEmittanceScan4_selector.triggerConditions = cms.vstring('HLT_L1AlwaysTrue_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsEmittanceScan2_datasetEmittanceScan5_selector
streamPhysicsEmittanceScan2_datasetEmittanceScan5_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsEmittanceScan2_datasetEmittanceScan5_selector.l1tResults = cms.InputTag('')
streamPhysicsEmittanceScan2_datasetEmittanceScan5_selector.throw      = cms.bool(False)
streamPhysicsEmittanceScan2_datasetEmittanceScan5_selector.triggerConditions = cms.vstring('HLT_L1AlwaysTrue_v1')


# stream PhysicsHLTPhysics0

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHLTPhysics0_datasetEphemeralHLTPhysics0_selector
streamPhysicsHLTPhysics0_datasetEphemeralHLTPhysics0_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHLTPhysics0_datasetEphemeralHLTPhysics0_selector.l1tResults = cms.InputTag('')
streamPhysicsHLTPhysics0_datasetEphemeralHLTPhysics0_selector.throw      = cms.bool(False)
streamPhysicsHLTPhysics0_datasetEphemeralHLTPhysics0_selector.triggerConditions = cms.vstring('HLT_EphemeralPhysics_v10')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHLTPhysics0_datasetEphemeralHLTPhysics1_selector
streamPhysicsHLTPhysics0_datasetEphemeralHLTPhysics1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHLTPhysics0_datasetEphemeralHLTPhysics1_selector.l1tResults = cms.InputTag('')
streamPhysicsHLTPhysics0_datasetEphemeralHLTPhysics1_selector.throw      = cms.bool(False)
streamPhysicsHLTPhysics0_datasetEphemeralHLTPhysics1_selector.triggerConditions = cms.vstring('HLT_EphemeralPhysics_v10')


# stream PhysicsHLTPhysics1

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHLTPhysics1_datasetEphemeralHLTPhysics2_selector
streamPhysicsHLTPhysics1_datasetEphemeralHLTPhysics2_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHLTPhysics1_datasetEphemeralHLTPhysics2_selector.l1tResults = cms.InputTag('')
streamPhysicsHLTPhysics1_datasetEphemeralHLTPhysics2_selector.throw      = cms.bool(False)
streamPhysicsHLTPhysics1_datasetEphemeralHLTPhysics2_selector.triggerConditions = cms.vstring('HLT_EphemeralPhysics_v10')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHLTPhysics1_datasetEphemeralHLTPhysics3_selector
streamPhysicsHLTPhysics1_datasetEphemeralHLTPhysics3_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHLTPhysics1_datasetEphemeralHLTPhysics3_selector.l1tResults = cms.InputTag('')
streamPhysicsHLTPhysics1_datasetEphemeralHLTPhysics3_selector.throw      = cms.bool(False)
streamPhysicsHLTPhysics1_datasetEphemeralHLTPhysics3_selector.triggerConditions = cms.vstring('HLT_EphemeralPhysics_v10')


# stream PhysicsHLTPhysics2

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHLTPhysics2_datasetEphemeralHLTPhysics4_selector
streamPhysicsHLTPhysics2_datasetEphemeralHLTPhysics4_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHLTPhysics2_datasetEphemeralHLTPhysics4_selector.l1tResults = cms.InputTag('')
streamPhysicsHLTPhysics2_datasetEphemeralHLTPhysics4_selector.throw      = cms.bool(False)
streamPhysicsHLTPhysics2_datasetEphemeralHLTPhysics4_selector.triggerConditions = cms.vstring('HLT_EphemeralPhysics_v10')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHLTPhysics2_datasetEphemeralHLTPhysics5_selector
streamPhysicsHLTPhysics2_datasetEphemeralHLTPhysics5_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHLTPhysics2_datasetEphemeralHLTPhysics5_selector.l1tResults = cms.InputTag('')
streamPhysicsHLTPhysics2_datasetEphemeralHLTPhysics5_selector.throw      = cms.bool(False)
streamPhysicsHLTPhysics2_datasetEphemeralHLTPhysics5_selector.triggerConditions = cms.vstring('HLT_EphemeralPhysics_v10')


# stream PhysicsHLTPhysics3

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHLTPhysics3_datasetEphemeralHLTPhysics6_selector
streamPhysicsHLTPhysics3_datasetEphemeralHLTPhysics6_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHLTPhysics3_datasetEphemeralHLTPhysics6_selector.l1tResults = cms.InputTag('')
streamPhysicsHLTPhysics3_datasetEphemeralHLTPhysics6_selector.throw      = cms.bool(False)
streamPhysicsHLTPhysics3_datasetEphemeralHLTPhysics6_selector.triggerConditions = cms.vstring('HLT_EphemeralPhysics_v10')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHLTPhysics3_datasetEphemeralHLTPhysics7_selector
streamPhysicsHLTPhysics3_datasetEphemeralHLTPhysics7_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHLTPhysics3_datasetEphemeralHLTPhysics7_selector.l1tResults = cms.InputTag('')
streamPhysicsHLTPhysics3_datasetEphemeralHLTPhysics7_selector.throw      = cms.bool(False)
streamPhysicsHLTPhysics3_datasetEphemeralHLTPhysics7_selector.triggerConditions = cms.vstring('HLT_EphemeralPhysics_v10')


# stream PhysicsJetMET0

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsJetMET0_datasetJetMET0_selector
streamPhysicsJetMET0_datasetJetMET0_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsJetMET0_datasetJetMET0_selector.l1tResults = cms.InputTag('')
streamPhysicsJetMET0_datasetJetMET0_selector.throw      = cms.bool(False)
streamPhysicsJetMET0_datasetJetMET0_selector.triggerConditions = cms.vstring(
    'HLT_AK8DiPFJet250_250_SoftDropMass40_v7',
    'HLT_AK8DiPFJet250_250_SoftDropMass50_v7',
    'HLT_AK8DiPFJet260_260_SoftDropMass30_v7',
    'HLT_AK8DiPFJet260_260_SoftDropMass40_v7',
    'HLT_AK8DiPFJet270_270_SoftDropMass30_v7',
    'HLT_AK8DiPFJet280_280_SoftDropMass30_v13',
    'HLT_AK8DiPFJet290_290_SoftDropMass30_v7',
    'HLT_AK8PFJet140_v28',
    'HLT_AK8PFJet200_v28',
    'HLT_AK8PFJet220_SoftDropMass40_PNetBB0p06_DoubleAK4PFJet60_30_PNet2BTagMean0p50_v10',
    'HLT_AK8PFJet220_SoftDropMass40_PNetBB0p06_DoubleAK4PFJet60_30_PNet2BTagMean0p53_v10',
    'HLT_AK8PFJet220_SoftDropMass40_PNetBB0p06_DoubleAK4PFJet60_30_PNet2BTagMean0p55_v10',
    'HLT_AK8PFJet220_SoftDropMass40_PNetBB0p06_DoubleAK4PFJet60_30_PNet2BTagMean0p60_v10',
    'HLT_AK8PFJet220_SoftDropMass40_v14',
    'HLT_AK8PFJet230_SoftDropMass40_PNetBB0p06_v10',
    'HLT_AK8PFJet230_SoftDropMass40_PNetBB0p10_v10',
    'HLT_AK8PFJet230_SoftDropMass40_PNetTauTau0p03_v10',
    'HLT_AK8PFJet230_SoftDropMass40_PNetTauTau0p05_v10',
    'HLT_AK8PFJet230_SoftDropMass40_v14',
    'HLT_AK8PFJet250_SoftDropMass40_PNetBB0p06_v10',
    'HLT_AK8PFJet250_SoftDropMass40_PNetBB0p10_v10',
    'HLT_AK8PFJet250_SoftDropMass40_PNetTauTau0p03_v10',
    'HLT_AK8PFJet250_SoftDropMass40_PNetTauTau0p05_v10',
    'HLT_AK8PFJet260_v29',
    'HLT_AK8PFJet275_Nch40_v7',
    'HLT_AK8PFJet275_Nch45_v7',
    'HLT_AK8PFJet275_SoftDropMass40_PNetBB0p06_v10',
    'HLT_AK8PFJet275_SoftDropMass40_PNetBB0p10_v10',
    'HLT_AK8PFJet275_SoftDropMass40_PNetTauTau0p03_v10',
    'HLT_AK8PFJet275_SoftDropMass40_PNetTauTau0p05_v10',
    'HLT_AK8PFJet320_v29',
    'HLT_AK8PFJet380_SoftDropMass30_v7',
    'HLT_AK8PFJet400_SoftDropMass30_v7',
    'HLT_AK8PFJet400_v29',
    'HLT_AK8PFJet40_v29',
    'HLT_AK8PFJet425_SoftDropMass30_v7',
    'HLT_AK8PFJet450_SoftDropMass30_v7',
    'HLT_AK8PFJet450_v29',
    'HLT_AK8PFJet500_v29',
    'HLT_AK8PFJet550_v24',
    'HLT_AK8PFJet60_v28',
    'HLT_AK8PFJet80_v29',
    'HLT_AK8PFJetFwd140_v27',
    'HLT_AK8PFJetFwd200_v27',
    'HLT_AK8PFJetFwd260_v28',
    'HLT_AK8PFJetFwd320_v28',
    'HLT_AK8PFJetFwd400_v28',
    'HLT_AK8PFJetFwd40_v28',
    'HLT_AK8PFJetFwd450_v28',
    'HLT_AK8PFJetFwd500_v28',
    'HLT_AK8PFJetFwd60_v27',
    'HLT_AK8PFJetFwd80_v27',
    'HLT_CaloJet500_NoJetID_v23',
    'HLT_CaloJet550_NoJetID_v18',
    'HLT_CaloMET350_NotCleaned_v15',
    'HLT_CaloMET90_NotCleaned_v15',
    'HLT_CaloMHT90_v15',
    'HLT_DiPFJetAve100_HFJEC_v30',
    'HLT_DiPFJetAve140_v26',
    'HLT_DiPFJetAve160_HFJEC_v29',
    'HLT_DiPFJetAve180_PPSMatch_Xi0p3_QuadJet_Max2ProtPerRP_v7',
    'HLT_DiPFJetAve200_v26',
    'HLT_DiPFJetAve220_HFJEC_v29',
    'HLT_DiPFJetAve260_HFJEC_v12',
    'HLT_DiPFJetAve260_v27',
    'HLT_DiPFJetAve300_HFJEC_v29',
    'HLT_DiPFJetAve320_v27',
    'HLT_DiPFJetAve400_v27',
    'HLT_DiPFJetAve40_v27',
    'HLT_DiPFJetAve500_v27',
    'HLT_DiPFJetAve60_HFJEC_v28',
    'HLT_DiPFJetAve60_v27',
    'HLT_DiPFJetAve80_HFJEC_v30',
    'HLT_DiPFJetAve80_v27',
    'HLT_DoublePFJets100_PNetBTag_0p11_v7',
    'HLT_DoublePFJets116MaxDeta1p6_PNet2BTag_0p11_v7',
    'HLT_DoublePFJets128MaxDeta1p6_PNet2BTag_0p11_v7',
    'HLT_DoublePFJets200_PNetBTag_0p11_v7',
    'HLT_DoublePFJets350_PNetBTag_0p11_v7',
    'HLT_DoublePFJets40_PNetBTag_0p11_v7',
    'HLT_HT350_v8',
    'HLT_HT425_v20',
    'HLT_L1ETMHadSeeds_v11',
    'HLT_L1Mu6HT240_v10',
    'HLT_MET105_IsoTrk50_v20',
    'HLT_MET120_IsoTrk50_v20',
    'HLT_Mu12_DoublePFJets100_PNetBTag_0p11_v7',
    'HLT_Mu12_DoublePFJets200_PNetBTag_0p11_v7',
    'HLT_Mu12_DoublePFJets350_PNetBTag_0p11_v7',
    'HLT_Mu12_DoublePFJets40MaxDeta1p6_PNet2BTag_0p11_v7',
    'HLT_Mu12_DoublePFJets40_PNetBTag_0p11_v7',
    'HLT_Mu12_DoublePFJets54MaxDeta1p6_PNet2BTag_0p11_v7',
    'HLT_Mu12eta2p3_PFJet40_v14',
    'HLT_Mu12eta2p3_v14',
    'HLT_PFHT1050_v31',
    'HLT_PFHT180_v30',
    'HLT_PFHT250_v30',
    'HLT_PFHT350_v32',
    'HLT_PFHT370_v30',
    'HLT_PFHT430_v30',
    'HLT_PFHT500_PFMET100_PFMHT100_IDTight_v25',
    'HLT_PFHT500_PFMET110_PFMHT110_IDTight_v25',
    'HLT_PFHT510_v30',
    'HLT_PFHT590_v30',
    'HLT_PFHT680_v30',
    'HLT_PFHT700_PFMET85_PFMHT85_IDTight_v25',
    'HLT_PFHT780_v30',
    'HLT_PFHT800_PFMET75_PFMHT75_IDTight_v25',
    'HLT_PFHT890_v30',
    'HLT_PFJet110_v13',
    'HLT_PFJet140_v32',
    'HLT_PFJet200_v32',
    'HLT_PFJet260_v33',
    'HLT_PFJet320_v33',
    'HLT_PFJet400_v33',
    'HLT_PFJet40_v34',
    'HLT_PFJet450_v34',
    'HLT_PFJet500_v34',
    'HLT_PFJet550_v24',
    'HLT_PFJet60_v34',
    'HLT_PFJet80_v34',
    'HLT_PFJetFwd140_v31',
    'HLT_PFJetFwd200_v31',
    'HLT_PFJetFwd260_v32',
    'HLT_PFJetFwd320_v32',
    'HLT_PFJetFwd400_v32',
    'HLT_PFJetFwd40_v32',
    'HLT_PFJetFwd450_v32',
    'HLT_PFJetFwd500_v32',
    'HLT_PFJetFwd60_v32',
    'HLT_PFJetFwd80_v31',
    'HLT_PFMET105_IsoTrk50_v14',
    'HLT_PFMET120_PFMHT120_IDTight_PFHT60_v22',
    'HLT_PFMET120_PFMHT120_IDTight_v33',
    'HLT_PFMET130_PFMHT130_IDTight_v33',
    'HLT_PFMET140_PFMHT140_IDTight_v33',
    'HLT_PFMET200_BeamHaloCleaned_v22',
    'HLT_PFMET200_NotCleaned_v22',
    'HLT_PFMET250_NotCleaned_v22',
    'HLT_PFMET300_NotCleaned_v22',
    'HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF_v13',
    'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_FilterHF_v13',
    'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v22',
    'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v33',
    'HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_FilterHF_v13',
    'HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v32',
    'HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_FilterHF_v13',
    'HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v32',
    'HLT_PFMETTypeOne140_PFMHT140_IDTight_v24',
    'HLT_PFMETTypeOne200_BeamHaloCleaned_v22'
)


# stream PhysicsJetMET1

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsJetMET1_datasetJetMET1_selector
streamPhysicsJetMET1_datasetJetMET1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsJetMET1_datasetJetMET1_selector.l1tResults = cms.InputTag('')
streamPhysicsJetMET1_datasetJetMET1_selector.throw      = cms.bool(False)
streamPhysicsJetMET1_datasetJetMET1_selector.triggerConditions = cms.vstring(
    'HLT_AK8DiPFJet250_250_SoftDropMass40_v7',
    'HLT_AK8DiPFJet250_250_SoftDropMass50_v7',
    'HLT_AK8DiPFJet260_260_SoftDropMass30_v7',
    'HLT_AK8DiPFJet260_260_SoftDropMass40_v7',
    'HLT_AK8DiPFJet270_270_SoftDropMass30_v7',
    'HLT_AK8DiPFJet280_280_SoftDropMass30_v13',
    'HLT_AK8DiPFJet290_290_SoftDropMass30_v7',
    'HLT_AK8PFJet140_v28',
    'HLT_AK8PFJet200_v28',
    'HLT_AK8PFJet220_SoftDropMass40_PNetBB0p06_DoubleAK4PFJet60_30_PNet2BTagMean0p50_v10',
    'HLT_AK8PFJet220_SoftDropMass40_PNetBB0p06_DoubleAK4PFJet60_30_PNet2BTagMean0p53_v10',
    'HLT_AK8PFJet220_SoftDropMass40_PNetBB0p06_DoubleAK4PFJet60_30_PNet2BTagMean0p55_v10',
    'HLT_AK8PFJet220_SoftDropMass40_PNetBB0p06_DoubleAK4PFJet60_30_PNet2BTagMean0p60_v10',
    'HLT_AK8PFJet220_SoftDropMass40_v14',
    'HLT_AK8PFJet230_SoftDropMass40_PNetBB0p06_v10',
    'HLT_AK8PFJet230_SoftDropMass40_PNetBB0p10_v10',
    'HLT_AK8PFJet230_SoftDropMass40_PNetTauTau0p03_v10',
    'HLT_AK8PFJet230_SoftDropMass40_PNetTauTau0p05_v10',
    'HLT_AK8PFJet230_SoftDropMass40_v14',
    'HLT_AK8PFJet250_SoftDropMass40_PNetBB0p06_v10',
    'HLT_AK8PFJet250_SoftDropMass40_PNetBB0p10_v10',
    'HLT_AK8PFJet250_SoftDropMass40_PNetTauTau0p03_v10',
    'HLT_AK8PFJet250_SoftDropMass40_PNetTauTau0p05_v10',
    'HLT_AK8PFJet260_v29',
    'HLT_AK8PFJet275_Nch40_v7',
    'HLT_AK8PFJet275_Nch45_v7',
    'HLT_AK8PFJet275_SoftDropMass40_PNetBB0p06_v10',
    'HLT_AK8PFJet275_SoftDropMass40_PNetBB0p10_v10',
    'HLT_AK8PFJet275_SoftDropMass40_PNetTauTau0p03_v10',
    'HLT_AK8PFJet275_SoftDropMass40_PNetTauTau0p05_v10',
    'HLT_AK8PFJet320_v29',
    'HLT_AK8PFJet380_SoftDropMass30_v7',
    'HLT_AK8PFJet400_SoftDropMass30_v7',
    'HLT_AK8PFJet400_v29',
    'HLT_AK8PFJet40_v29',
    'HLT_AK8PFJet425_SoftDropMass30_v7',
    'HLT_AK8PFJet450_SoftDropMass30_v7',
    'HLT_AK8PFJet450_v29',
    'HLT_AK8PFJet500_v29',
    'HLT_AK8PFJet550_v24',
    'HLT_AK8PFJet60_v28',
    'HLT_AK8PFJet80_v29',
    'HLT_AK8PFJetFwd140_v27',
    'HLT_AK8PFJetFwd200_v27',
    'HLT_AK8PFJetFwd260_v28',
    'HLT_AK8PFJetFwd320_v28',
    'HLT_AK8PFJetFwd400_v28',
    'HLT_AK8PFJetFwd40_v28',
    'HLT_AK8PFJetFwd450_v28',
    'HLT_AK8PFJetFwd500_v28',
    'HLT_AK8PFJetFwd60_v27',
    'HLT_AK8PFJetFwd80_v27',
    'HLT_CaloJet500_NoJetID_v23',
    'HLT_CaloJet550_NoJetID_v18',
    'HLT_CaloMET350_NotCleaned_v15',
    'HLT_CaloMET90_NotCleaned_v15',
    'HLT_CaloMHT90_v15',
    'HLT_DiPFJetAve100_HFJEC_v30',
    'HLT_DiPFJetAve140_v26',
    'HLT_DiPFJetAve160_HFJEC_v29',
    'HLT_DiPFJetAve180_PPSMatch_Xi0p3_QuadJet_Max2ProtPerRP_v7',
    'HLT_DiPFJetAve200_v26',
    'HLT_DiPFJetAve220_HFJEC_v29',
    'HLT_DiPFJetAve260_HFJEC_v12',
    'HLT_DiPFJetAve260_v27',
    'HLT_DiPFJetAve300_HFJEC_v29',
    'HLT_DiPFJetAve320_v27',
    'HLT_DiPFJetAve400_v27',
    'HLT_DiPFJetAve40_v27',
    'HLT_DiPFJetAve500_v27',
    'HLT_DiPFJetAve60_HFJEC_v28',
    'HLT_DiPFJetAve60_v27',
    'HLT_DiPFJetAve80_HFJEC_v30',
    'HLT_DiPFJetAve80_v27',
    'HLT_DoublePFJets100_PNetBTag_0p11_v7',
    'HLT_DoublePFJets116MaxDeta1p6_PNet2BTag_0p11_v7',
    'HLT_DoublePFJets128MaxDeta1p6_PNet2BTag_0p11_v7',
    'HLT_DoublePFJets200_PNetBTag_0p11_v7',
    'HLT_DoublePFJets350_PNetBTag_0p11_v7',
    'HLT_DoublePFJets40_PNetBTag_0p11_v7',
    'HLT_HT350_v8',
    'HLT_HT425_v20',
    'HLT_L1ETMHadSeeds_v11',
    'HLT_L1Mu6HT240_v10',
    'HLT_MET105_IsoTrk50_v20',
    'HLT_MET120_IsoTrk50_v20',
    'HLT_Mu12_DoublePFJets100_PNetBTag_0p11_v7',
    'HLT_Mu12_DoublePFJets200_PNetBTag_0p11_v7',
    'HLT_Mu12_DoublePFJets350_PNetBTag_0p11_v7',
    'HLT_Mu12_DoublePFJets40MaxDeta1p6_PNet2BTag_0p11_v7',
    'HLT_Mu12_DoublePFJets40_PNetBTag_0p11_v7',
    'HLT_Mu12_DoublePFJets54MaxDeta1p6_PNet2BTag_0p11_v7',
    'HLT_Mu12eta2p3_PFJet40_v14',
    'HLT_Mu12eta2p3_v14',
    'HLT_PFHT1050_v31',
    'HLT_PFHT180_v30',
    'HLT_PFHT250_v30',
    'HLT_PFHT350_v32',
    'HLT_PFHT370_v30',
    'HLT_PFHT430_v30',
    'HLT_PFHT500_PFMET100_PFMHT100_IDTight_v25',
    'HLT_PFHT500_PFMET110_PFMHT110_IDTight_v25',
    'HLT_PFHT510_v30',
    'HLT_PFHT590_v30',
    'HLT_PFHT680_v30',
    'HLT_PFHT700_PFMET85_PFMHT85_IDTight_v25',
    'HLT_PFHT780_v30',
    'HLT_PFHT800_PFMET75_PFMHT75_IDTight_v25',
    'HLT_PFHT890_v30',
    'HLT_PFJet110_v13',
    'HLT_PFJet140_v32',
    'HLT_PFJet200_v32',
    'HLT_PFJet260_v33',
    'HLT_PFJet320_v33',
    'HLT_PFJet400_v33',
    'HLT_PFJet40_v34',
    'HLT_PFJet450_v34',
    'HLT_PFJet500_v34',
    'HLT_PFJet550_v24',
    'HLT_PFJet60_v34',
    'HLT_PFJet80_v34',
    'HLT_PFJetFwd140_v31',
    'HLT_PFJetFwd200_v31',
    'HLT_PFJetFwd260_v32',
    'HLT_PFJetFwd320_v32',
    'HLT_PFJetFwd400_v32',
    'HLT_PFJetFwd40_v32',
    'HLT_PFJetFwd450_v32',
    'HLT_PFJetFwd500_v32',
    'HLT_PFJetFwd60_v32',
    'HLT_PFJetFwd80_v31',
    'HLT_PFMET105_IsoTrk50_v14',
    'HLT_PFMET120_PFMHT120_IDTight_PFHT60_v22',
    'HLT_PFMET120_PFMHT120_IDTight_v33',
    'HLT_PFMET130_PFMHT130_IDTight_v33',
    'HLT_PFMET140_PFMHT140_IDTight_v33',
    'HLT_PFMET200_BeamHaloCleaned_v22',
    'HLT_PFMET200_NotCleaned_v22',
    'HLT_PFMET250_NotCleaned_v22',
    'HLT_PFMET300_NotCleaned_v22',
    'HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF_v13',
    'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_FilterHF_v13',
    'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v22',
    'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v33',
    'HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_FilterHF_v13',
    'HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v32',
    'HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_FilterHF_v13',
    'HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v32',
    'HLT_PFMETTypeOne140_PFMHT140_IDTight_v24',
    'HLT_PFMETTypeOne200_BeamHaloCleaned_v22'
)


# stream PhysicsMuon0

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsMuon0_datasetMuon0_selector
streamPhysicsMuon0_datasetMuon0_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsMuon0_datasetMuon0_selector.l1tResults = cms.InputTag('')
streamPhysicsMuon0_datasetMuon0_selector.throw      = cms.bool(False)
streamPhysicsMuon0_datasetMuon0_selector.triggerConditions = cms.vstring(
    'HLT_CascadeMu100_v14',
    'HLT_CscCluster100_Ele5_v5',
    'HLT_CscCluster100_Mu5_v7',
    'HLT_CscCluster100_PNetTauhPFJet10_Loose_v7',
    'HLT_CscCluster50_Photon20Unseeded_v5',
    'HLT_CscCluster50_Photon30Unseeded_v5',
    'HLT_CscCluster_Loose_v11',
    'HLT_CscCluster_Medium_v11',
    'HLT_CscCluster_Tight_v11',
    'HLT_DisplacedMu24_MediumChargedIsoDisplacedPFTauHPS24_v9',
    'HLT_DoubleCscCluster100_v8',
    'HLT_DoubleCscCluster75_v8',
    'HLT_DoubleL2Mu10NoVtx_2Cha_CosmicSeed_VetoL3Mu0DxyMax1cm_v12',
    'HLT_DoubleL2Mu10NoVtx_2Cha_VetoL3Mu0DxyMax1cm_v13',
    'HLT_DoubleL2Mu12NoVtx_2Cha_CosmicSeed_VetoL3Mu0DxyMax1cm_v12',
    'HLT_DoubleL2Mu12NoVtx_2Cha_VetoL3Mu0DxyMax1cm_v12',
    'HLT_DoubleL2Mu14NoVtx_2Cha_VetoL3Mu0DxyMax1cm_v12',
    'HLT_DoubleL2Mu23NoVtx_2Cha_CosmicSeed_v12',
    'HLT_DoubleL2Mu23NoVtx_2Cha_v12',
    'HLT_DoubleL2Mu25NoVtx_2Cha_CosmicSeed_Eta2p4_v12',
    'HLT_DoubleL2Mu25NoVtx_2Cha_CosmicSeed_v12',
    'HLT_DoubleL2Mu25NoVtx_2Cha_Eta2p4_v12',
    'HLT_DoubleL2Mu25NoVtx_2Cha_v12',
    'HLT_DoubleL2Mu30NoVtx_2Cha_CosmicSeed_Eta2p4_v12',
    'HLT_DoubleL2Mu30NoVtx_2Cha_Eta2p4_v12',
    'HLT_DoubleL2Mu50_v12',
    'HLT_DoubleL2Mu_L3Mu16NoVtx_VetoL3Mu0DxyMax0p1cm_v11',
    'HLT_DoubleL2Mu_L3Mu18NoVtx_VetoL3Mu0DxyMax0p1cm_v11',
    'HLT_DoubleL3Mu16_10NoVtx_DxyMin0p01cm_v12',
    'HLT_DoubleL3Mu18_10NoVtx_DxyMin0p01cm_v11',
    'HLT_DoubleL3Mu20_10NoVtx_DxyMin0p01cm_v11',
    'HLT_DoubleL3dTksMu16_10NoVtx_DxyMin0p01cm_v11',
    'HLT_DoubleMu3_DCA_PFMET50_PFMHT60_Mass2p0_noDCA_v7',
    'HLT_DoubleMu3_DCA_PFMET50_PFMHT60_Mass2p0_v7',
    'HLT_DoubleMu3_DCA_PFMET50_PFMHT60_v23',
    'HLT_DoubleMu3_DZ_PFMET50_PFMHT60_v23',
    'HLT_DoubleMu3_DZ_PFMET70_PFMHT70_v23',
    'HLT_DoubleMu3_DZ_PFMET90_PFMHT90_v23',
    'HLT_DoubleMu43NoFiltersNoVtx_v15',
    'HLT_DoubleMu48NoFiltersNoVtx_v15',
    'HLT_DoubleMu4_Mass3p8_DZ_PFHT350_v21',
    'HLT_HighPtTkMu100_v13',
    'HLT_IsoMu20_eta2p1_PNetTauhPFJet27_Loose_eta2p3_CrossL1_v7',
    'HLT_IsoMu20_eta2p1_PNetTauhPFJet27_Medium_eta2p3_CrossL1_v7',
    'HLT_IsoMu20_eta2p1_PNetTauhPFJet27_Tight_eta2p3_CrossL1_v7',
    'HLT_IsoMu20_v28',
    'HLT_IsoMu24_OneProng32_v10',
    'HLT_IsoMu24_TwoProngs35_v14',
    'HLT_IsoMu24_eta2p1_L1HT200_QuadPFJet25_PNet1Tauh0p50_v1',
    'HLT_IsoMu24_eta2p1_L1HT200_QuadPFJet25_v1',
    'HLT_IsoMu24_eta2p1_L1HT200_v1',
    'HLT_IsoMu24_eta2p1_PFHT250_QuadPFJet25_PNet1Tauh0p50_v7',
    'HLT_IsoMu24_eta2p1_PFHT250_QuadPFJet25_v7',
    'HLT_IsoMu24_eta2p1_PFHT250_v7',
    'HLT_IsoMu24_eta2p1_PNetTauhPFJet130_Loose_L2NN_eta2p3_CrossL1_v7',
    'HLT_IsoMu24_eta2p1_PNetTauhPFJet130_Medium_L2NN_eta2p3_CrossL1_v7',
    'HLT_IsoMu24_eta2p1_PNetTauhPFJet130_Tight_L2NN_eta2p3_CrossL1_v7',
    'HLT_IsoMu24_eta2p1_PNetTauhPFJet20_eta2p2_SingleL1_v7',
    'HLT_IsoMu24_eta2p1_PNetTauhPFJet26_L2NN_eta2p3_CrossL1_PFJet60_v7',
    'HLT_IsoMu24_eta2p1_PNetTauhPFJet26_L2NN_eta2p3_CrossL1_PFJet75_v7',
    'HLT_IsoMu24_eta2p1_PNetTauhPFJet26_L2NN_eta2p3_CrossL1_v7',
    'HLT_IsoMu24_eta2p1_PNetTauhPFJet30_Loose_eta2p3_CrossL1_ETau_Monitoring_v7',
    'HLT_IsoMu24_eta2p1_PNetTauhPFJet30_Medium_L2NN_eta2p3_CrossL1_v7',
    'HLT_IsoMu24_eta2p1_PNetTauhPFJet30_Medium_eta2p3_CrossL1_ETau_Monitoring_v7',
    'HLT_IsoMu24_eta2p1_PNetTauhPFJet30_Tight_L2NN_eta2p3_CrossL1_v7',
    'HLT_IsoMu24_eta2p1_PNetTauhPFJet30_Tight_eta2p3_CrossL1_ETau_Monitoring_v7',
    'HLT_IsoMu24_eta2p1_PNetTauhPFJet45_L2NN_eta2p3_CrossL1_v7',
    'HLT_IsoMu24_eta2p1_SinglePFJet25_PNet1Tauh0p50_v7',
    'HLT_IsoMu24_eta2p1_v28',
    'HLT_IsoMu24_v26',
    'HLT_IsoMu27_MediumChargedIsoDisplacedPFTauHPS24_eta2p1_SingleL1_v9',
    'HLT_IsoMu27_v29',
    'HLT_IsoMu50_AK8PFJet220_SoftDropMass40_PNetBB0p06_v10',
    'HLT_IsoMu50_AK8PFJet220_SoftDropMass40_v13',
    'HLT_IsoMu50_AK8PFJet230_SoftDropMass40_PNetBB0p06_v10',
    'HLT_IsoMu50_AK8PFJet230_SoftDropMass40_PNetBB0p10_v10',
    'HLT_IsoMu50_AK8PFJet230_SoftDropMass40_v13',
    'HLT_IsoTrk200_L1SingleMuShower_v5',
    'HLT_IsoTrk400_L1SingleMuShower_v5',
    'HLT_L1CSCShower_DTCluster50_v11',
    'HLT_L1CSCShower_DTCluster75_v11',
    'HLT_L2Mu50NoVtx_3Cha_CosmicSeed_VetoL3Mu0DxyMax1cm_v5',
    'HLT_L2Mu50NoVtx_3Cha_VetoL3Mu0DxyMax1cm_v5',
    'HLT_L3Mu30NoVtx_DxyMin0p01cm_v4',
    'HLT_L3Mu50NoVtx_DxyMin0p01cm_v4',
    'HLT_L3dTksMu10_NoVtx_DxyMin0p01cm_v11',
    'HLT_Mu12_IsoVVL_PFHT150_PNetBTag0p53_v6',
    'HLT_Mu15_IsoVVVL_PFHT450_PFMET50_v28',
    'HLT_Mu15_IsoVVVL_PFHT450_v28',
    'HLT_Mu15_IsoVVVL_PFHT600_v32',
    'HLT_Mu15_v16',
    'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8_v18',
    'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8_AK8CaloJet30_v6',
    'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8_AK8PFJet30_v6',
    'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8_CaloJet30_v6',
    'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8_PFJet30_v7',
    'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8_v18',
    'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v28',
    'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_v27',
    'HLT_Mu17_TrkIsoVVL_v26',
    'HLT_Mu17_v26',
    'HLT_Mu18_Mu9_SameSign_v17',
    'HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ_Mass3p8_v16',
    'HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ_Mass8_v16',
    'HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ_v16',
    'HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_v16',
    'HLT_Mu19_TrkIsoVVL_v17',
    'HLT_Mu19_v17',
    'HLT_Mu20_v25',
    'HLT_Mu27_v26',
    'HLT_Mu37_TkMu27_v18',
    'HLT_Mu3_PFJet40_v29',
    'HLT_Mu3er1p5_PFJet100er2p5_PFMET100_PFMHT100_IDTight_v15',
    'HLT_Mu3er1p5_PFJet100er2p5_PFMET80_PFMHT80_IDTight_v15',
    'HLT_Mu3er1p5_PFJet100er2p5_PFMET90_PFMHT90_IDTight_v15',
    'HLT_Mu3er1p5_PFJet100er2p5_PFMETNoMu100_PFMHTNoMu100_IDTight_v15',
    'HLT_Mu3er1p5_PFJet100er2p5_PFMETNoMu80_PFMHTNoMu80_IDTight_v15',
    'HLT_Mu3er1p5_PFJet100er2p5_PFMETNoMu90_PFMHTNoMu90_IDTight_v15',
    'HLT_Mu50_IsoVVVL_PFHT450_v28',
    'HLT_Mu50_L1SingleMuShower_v12',
    'HLT_Mu50_v26',
    'HLT_Mu55_v16',
    'HLT_Mu8_TrkIsoVVL_v25',
    'HLT_Mu8_v25',
    'HLT_TripleMu_10_5_5_DZ_v23',
    'HLT_TripleMu_12_10_5_v23',
    'HLT_TripleMu_5_3_3_Mass3p8_DCA_v16',
    'HLT_TripleMu_5_3_3_Mass3p8_DZ_v21',
    'HLT_TrkMu12_DoubleTrkMu5NoFiltersNoVtx_v18'
)


# stream PhysicsMuon1

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsMuon1_datasetMuon1_selector
streamPhysicsMuon1_datasetMuon1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsMuon1_datasetMuon1_selector.l1tResults = cms.InputTag('')
streamPhysicsMuon1_datasetMuon1_selector.throw      = cms.bool(False)
streamPhysicsMuon1_datasetMuon1_selector.triggerConditions = cms.vstring(
    'HLT_CascadeMu100_v14',
    'HLT_CscCluster100_Ele5_v5',
    'HLT_CscCluster100_Mu5_v7',
    'HLT_CscCluster100_PNetTauhPFJet10_Loose_v7',
    'HLT_CscCluster50_Photon20Unseeded_v5',
    'HLT_CscCluster50_Photon30Unseeded_v5',
    'HLT_CscCluster_Loose_v11',
    'HLT_CscCluster_Medium_v11',
    'HLT_CscCluster_Tight_v11',
    'HLT_DisplacedMu24_MediumChargedIsoDisplacedPFTauHPS24_v9',
    'HLT_DoubleCscCluster100_v8',
    'HLT_DoubleCscCluster75_v8',
    'HLT_DoubleL2Mu10NoVtx_2Cha_CosmicSeed_VetoL3Mu0DxyMax1cm_v12',
    'HLT_DoubleL2Mu10NoVtx_2Cha_VetoL3Mu0DxyMax1cm_v13',
    'HLT_DoubleL2Mu12NoVtx_2Cha_CosmicSeed_VetoL3Mu0DxyMax1cm_v12',
    'HLT_DoubleL2Mu12NoVtx_2Cha_VetoL3Mu0DxyMax1cm_v12',
    'HLT_DoubleL2Mu14NoVtx_2Cha_VetoL3Mu0DxyMax1cm_v12',
    'HLT_DoubleL2Mu23NoVtx_2Cha_CosmicSeed_v12',
    'HLT_DoubleL2Mu23NoVtx_2Cha_v12',
    'HLT_DoubleL2Mu25NoVtx_2Cha_CosmicSeed_Eta2p4_v12',
    'HLT_DoubleL2Mu25NoVtx_2Cha_CosmicSeed_v12',
    'HLT_DoubleL2Mu25NoVtx_2Cha_Eta2p4_v12',
    'HLT_DoubleL2Mu25NoVtx_2Cha_v12',
    'HLT_DoubleL2Mu30NoVtx_2Cha_CosmicSeed_Eta2p4_v12',
    'HLT_DoubleL2Mu30NoVtx_2Cha_Eta2p4_v12',
    'HLT_DoubleL2Mu50_v12',
    'HLT_DoubleL2Mu_L3Mu16NoVtx_VetoL3Mu0DxyMax0p1cm_v11',
    'HLT_DoubleL2Mu_L3Mu18NoVtx_VetoL3Mu0DxyMax0p1cm_v11',
    'HLT_DoubleL3Mu16_10NoVtx_DxyMin0p01cm_v12',
    'HLT_DoubleL3Mu18_10NoVtx_DxyMin0p01cm_v11',
    'HLT_DoubleL3Mu20_10NoVtx_DxyMin0p01cm_v11',
    'HLT_DoubleL3dTksMu16_10NoVtx_DxyMin0p01cm_v11',
    'HLT_DoubleMu3_DCA_PFMET50_PFMHT60_Mass2p0_noDCA_v7',
    'HLT_DoubleMu3_DCA_PFMET50_PFMHT60_Mass2p0_v7',
    'HLT_DoubleMu3_DCA_PFMET50_PFMHT60_v23',
    'HLT_DoubleMu3_DZ_PFMET50_PFMHT60_v23',
    'HLT_DoubleMu3_DZ_PFMET70_PFMHT70_v23',
    'HLT_DoubleMu3_DZ_PFMET90_PFMHT90_v23',
    'HLT_DoubleMu43NoFiltersNoVtx_v15',
    'HLT_DoubleMu48NoFiltersNoVtx_v15',
    'HLT_DoubleMu4_Mass3p8_DZ_PFHT350_v21',
    'HLT_HighPtTkMu100_v13',
    'HLT_IsoMu20_eta2p1_PNetTauhPFJet27_Loose_eta2p3_CrossL1_v7',
    'HLT_IsoMu20_eta2p1_PNetTauhPFJet27_Medium_eta2p3_CrossL1_v7',
    'HLT_IsoMu20_eta2p1_PNetTauhPFJet27_Tight_eta2p3_CrossL1_v7',
    'HLT_IsoMu20_v28',
    'HLT_IsoMu24_OneProng32_v10',
    'HLT_IsoMu24_TwoProngs35_v14',
    'HLT_IsoMu24_eta2p1_L1HT200_QuadPFJet25_PNet1Tauh0p50_v1',
    'HLT_IsoMu24_eta2p1_L1HT200_QuadPFJet25_v1',
    'HLT_IsoMu24_eta2p1_L1HT200_v1',
    'HLT_IsoMu24_eta2p1_PFHT250_QuadPFJet25_PNet1Tauh0p50_v7',
    'HLT_IsoMu24_eta2p1_PFHT250_QuadPFJet25_v7',
    'HLT_IsoMu24_eta2p1_PFHT250_v7',
    'HLT_IsoMu24_eta2p1_PNetTauhPFJet130_Loose_L2NN_eta2p3_CrossL1_v7',
    'HLT_IsoMu24_eta2p1_PNetTauhPFJet130_Medium_L2NN_eta2p3_CrossL1_v7',
    'HLT_IsoMu24_eta2p1_PNetTauhPFJet130_Tight_L2NN_eta2p3_CrossL1_v7',
    'HLT_IsoMu24_eta2p1_PNetTauhPFJet20_eta2p2_SingleL1_v7',
    'HLT_IsoMu24_eta2p1_PNetTauhPFJet26_L2NN_eta2p3_CrossL1_PFJet60_v7',
    'HLT_IsoMu24_eta2p1_PNetTauhPFJet26_L2NN_eta2p3_CrossL1_PFJet75_v7',
    'HLT_IsoMu24_eta2p1_PNetTauhPFJet26_L2NN_eta2p3_CrossL1_v7',
    'HLT_IsoMu24_eta2p1_PNetTauhPFJet30_Loose_eta2p3_CrossL1_ETau_Monitoring_v7',
    'HLT_IsoMu24_eta2p1_PNetTauhPFJet30_Medium_L2NN_eta2p3_CrossL1_v7',
    'HLT_IsoMu24_eta2p1_PNetTauhPFJet30_Medium_eta2p3_CrossL1_ETau_Monitoring_v7',
    'HLT_IsoMu24_eta2p1_PNetTauhPFJet30_Tight_L2NN_eta2p3_CrossL1_v7',
    'HLT_IsoMu24_eta2p1_PNetTauhPFJet30_Tight_eta2p3_CrossL1_ETau_Monitoring_v7',
    'HLT_IsoMu24_eta2p1_PNetTauhPFJet45_L2NN_eta2p3_CrossL1_v7',
    'HLT_IsoMu24_eta2p1_SinglePFJet25_PNet1Tauh0p50_v7',
    'HLT_IsoMu24_eta2p1_v28',
    'HLT_IsoMu24_v26',
    'HLT_IsoMu27_MediumChargedIsoDisplacedPFTauHPS24_eta2p1_SingleL1_v9',
    'HLT_IsoMu27_v29',
    'HLT_IsoMu50_AK8PFJet220_SoftDropMass40_PNetBB0p06_v10',
    'HLT_IsoMu50_AK8PFJet220_SoftDropMass40_v13',
    'HLT_IsoMu50_AK8PFJet230_SoftDropMass40_PNetBB0p06_v10',
    'HLT_IsoMu50_AK8PFJet230_SoftDropMass40_PNetBB0p10_v10',
    'HLT_IsoMu50_AK8PFJet230_SoftDropMass40_v13',
    'HLT_IsoTrk200_L1SingleMuShower_v5',
    'HLT_IsoTrk400_L1SingleMuShower_v5',
    'HLT_L1CSCShower_DTCluster50_v11',
    'HLT_L1CSCShower_DTCluster75_v11',
    'HLT_L2Mu50NoVtx_3Cha_CosmicSeed_VetoL3Mu0DxyMax1cm_v5',
    'HLT_L2Mu50NoVtx_3Cha_VetoL3Mu0DxyMax1cm_v5',
    'HLT_L3Mu30NoVtx_DxyMin0p01cm_v4',
    'HLT_L3Mu50NoVtx_DxyMin0p01cm_v4',
    'HLT_L3dTksMu10_NoVtx_DxyMin0p01cm_v11',
    'HLT_Mu12_IsoVVL_PFHT150_PNetBTag0p53_v6',
    'HLT_Mu15_IsoVVVL_PFHT450_PFMET50_v28',
    'HLT_Mu15_IsoVVVL_PFHT450_v28',
    'HLT_Mu15_IsoVVVL_PFHT600_v32',
    'HLT_Mu15_v16',
    'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8_v18',
    'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8_AK8CaloJet30_v6',
    'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8_AK8PFJet30_v6',
    'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8_CaloJet30_v6',
    'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8_PFJet30_v7',
    'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8_v18',
    'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v28',
    'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_v27',
    'HLT_Mu17_TrkIsoVVL_v26',
    'HLT_Mu17_v26',
    'HLT_Mu18_Mu9_SameSign_v17',
    'HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ_Mass3p8_v16',
    'HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ_Mass8_v16',
    'HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ_v16',
    'HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_v16',
    'HLT_Mu19_TrkIsoVVL_v17',
    'HLT_Mu19_v17',
    'HLT_Mu20_v25',
    'HLT_Mu27_v26',
    'HLT_Mu37_TkMu27_v18',
    'HLT_Mu3_PFJet40_v29',
    'HLT_Mu3er1p5_PFJet100er2p5_PFMET100_PFMHT100_IDTight_v15',
    'HLT_Mu3er1p5_PFJet100er2p5_PFMET80_PFMHT80_IDTight_v15',
    'HLT_Mu3er1p5_PFJet100er2p5_PFMET90_PFMHT90_IDTight_v15',
    'HLT_Mu3er1p5_PFJet100er2p5_PFMETNoMu100_PFMHTNoMu100_IDTight_v15',
    'HLT_Mu3er1p5_PFJet100er2p5_PFMETNoMu80_PFMHTNoMu80_IDTight_v15',
    'HLT_Mu3er1p5_PFJet100er2p5_PFMETNoMu90_PFMHTNoMu90_IDTight_v15',
    'HLT_Mu50_IsoVVVL_PFHT450_v28',
    'HLT_Mu50_L1SingleMuShower_v12',
    'HLT_Mu50_v26',
    'HLT_Mu55_v16',
    'HLT_Mu8_TrkIsoVVL_v25',
    'HLT_Mu8_v25',
    'HLT_TripleMu_10_5_5_DZ_v23',
    'HLT_TripleMu_12_10_5_v23',
    'HLT_TripleMu_5_3_3_Mass3p8_DCA_v16',
    'HLT_TripleMu_5_3_3_Mass3p8_DZ_v21',
    'HLT_TrkMu12_DoubleTrkMu5NoFiltersNoVtx_v18'
)


# stream PhysicsScoutingPFMonitor

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsScoutingPFMonitor_datasetScoutingPFMonitor_selector
streamPhysicsScoutingPFMonitor_datasetScoutingPFMonitor_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsScoutingPFMonitor_datasetScoutingPFMonitor_selector.l1tResults = cms.InputTag('')
streamPhysicsScoutingPFMonitor_datasetScoutingPFMonitor_selector.throw      = cms.bool(False)
streamPhysicsScoutingPFMonitor_datasetScoutingPFMonitor_selector.triggerConditions = cms.vstring(
    'DST_PFScouting_AXOLoose_v5',
    'DST_PFScouting_AXOMedium_v1',
    'DST_PFScouting_AXOTight_v7',
    'DST_PFScouting_AXOVLoose_v5',
    'DST_PFScouting_AXOVTight_v5',
    'DST_PFScouting_CICADALoose_v3',
    'DST_PFScouting_CICADAMedium_v3',
    'DST_PFScouting_CICADATight_v3',
    'DST_PFScouting_CICADAVLoose_v3',
    'DST_PFScouting_CICADAVTight_v3',
    'DST_PFScouting_DoubleEG_v7',
    'DST_PFScouting_DoubleMuonNoVtx_v1',
    'DST_PFScouting_DoubleMuonVtxMonitorJPsi_v1',
    'DST_PFScouting_DoubleMuonVtxMonitorZ_v1',
    'DST_PFScouting_DoubleMuonVtx_v1',
    'DST_PFScouting_JetHT_v7',
    'DST_PFScouting_SingleMuon_v7',
    'DST_PFScouting_SinglePhotonEB_v4',
    'DST_PFScouting_ZeroBias_v5'
)


# stream PhysicsZeroBias0

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsZeroBias0_datasetEphemeralZeroBias0_selector
streamPhysicsZeroBias0_datasetEphemeralZeroBias0_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsZeroBias0_datasetEphemeralZeroBias0_selector.l1tResults = cms.InputTag('')
streamPhysicsZeroBias0_datasetEphemeralZeroBias0_selector.throw      = cms.bool(False)
streamPhysicsZeroBias0_datasetEphemeralZeroBias0_selector.triggerConditions = cms.vstring('HLT_EphemeralZeroBias_v10')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsZeroBias0_datasetEphemeralZeroBias1_selector
streamPhysicsZeroBias0_datasetEphemeralZeroBias1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsZeroBias0_datasetEphemeralZeroBias1_selector.l1tResults = cms.InputTag('')
streamPhysicsZeroBias0_datasetEphemeralZeroBias1_selector.throw      = cms.bool(False)
streamPhysicsZeroBias0_datasetEphemeralZeroBias1_selector.triggerConditions = cms.vstring('HLT_EphemeralZeroBias_v10')


# stream PhysicsZeroBias1

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsZeroBias1_datasetEphemeralZeroBias2_selector
streamPhysicsZeroBias1_datasetEphemeralZeroBias2_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsZeroBias1_datasetEphemeralZeroBias2_selector.l1tResults = cms.InputTag('')
streamPhysicsZeroBias1_datasetEphemeralZeroBias2_selector.throw      = cms.bool(False)
streamPhysicsZeroBias1_datasetEphemeralZeroBias2_selector.triggerConditions = cms.vstring('HLT_EphemeralZeroBias_v10')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsZeroBias1_datasetEphemeralZeroBias3_selector
streamPhysicsZeroBias1_datasetEphemeralZeroBias3_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsZeroBias1_datasetEphemeralZeroBias3_selector.l1tResults = cms.InputTag('')
streamPhysicsZeroBias1_datasetEphemeralZeroBias3_selector.throw      = cms.bool(False)
streamPhysicsZeroBias1_datasetEphemeralZeroBias3_selector.triggerConditions = cms.vstring('HLT_EphemeralZeroBias_v10')


# stream PhysicsZeroBias2

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsZeroBias2_datasetEphemeralZeroBias4_selector
streamPhysicsZeroBias2_datasetEphemeralZeroBias4_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsZeroBias2_datasetEphemeralZeroBias4_selector.l1tResults = cms.InputTag('')
streamPhysicsZeroBias2_datasetEphemeralZeroBias4_selector.throw      = cms.bool(False)
streamPhysicsZeroBias2_datasetEphemeralZeroBias4_selector.triggerConditions = cms.vstring('HLT_EphemeralZeroBias_v10')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsZeroBias2_datasetEphemeralZeroBias5_selector
streamPhysicsZeroBias2_datasetEphemeralZeroBias5_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsZeroBias2_datasetEphemeralZeroBias5_selector.l1tResults = cms.InputTag('')
streamPhysicsZeroBias2_datasetEphemeralZeroBias5_selector.throw      = cms.bool(False)
streamPhysicsZeroBias2_datasetEphemeralZeroBias5_selector.triggerConditions = cms.vstring('HLT_EphemeralZeroBias_v10')


# stream PhysicsZeroBias3

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsZeroBias3_datasetEphemeralZeroBias6_selector
streamPhysicsZeroBias3_datasetEphemeralZeroBias6_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsZeroBias3_datasetEphemeralZeroBias6_selector.l1tResults = cms.InputTag('')
streamPhysicsZeroBias3_datasetEphemeralZeroBias6_selector.throw      = cms.bool(False)
streamPhysicsZeroBias3_datasetEphemeralZeroBias6_selector.triggerConditions = cms.vstring('HLT_EphemeralZeroBias_v10')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsZeroBias3_datasetEphemeralZeroBias7_selector
streamPhysicsZeroBias3_datasetEphemeralZeroBias7_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsZeroBias3_datasetEphemeralZeroBias7_selector.l1tResults = cms.InputTag('')
streamPhysicsZeroBias3_datasetEphemeralZeroBias7_selector.throw      = cms.bool(False)
streamPhysicsZeroBias3_datasetEphemeralZeroBias7_selector.triggerConditions = cms.vstring('HLT_EphemeralZeroBias_v10')

