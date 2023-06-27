# /dev/CMSSW_13_0_0/GRun

import FWCore.ParameterSet.Config as cms


# stream ParkingDoubleElectronLowMass

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingDoubleElectronLowMass_datasetParkingDoubleElectronLowMass_selector
streamParkingDoubleElectronLowMass_datasetParkingDoubleElectronLowMass_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingDoubleElectronLowMass_datasetParkingDoubleElectronLowMass_selector.l1tResults = cms.InputTag('')
streamParkingDoubleElectronLowMass_datasetParkingDoubleElectronLowMass_selector.throw      = cms.bool(False)
streamParkingDoubleElectronLowMass_datasetParkingDoubleElectronLowMass_selector.triggerConditions = cms.vstring(
    'HLT_DoubleEle10_eta1p22_mMax6_dz0p8_v3',
    'HLT_DoubleEle10_eta1p22_mMax6_trkHits10_v3',
    'HLT_DoubleEle10_eta1p22_mMax6_v4',
    'HLT_DoubleEle4_eta1p22_mMax6_dz0p8_v3',
    'HLT_DoubleEle4_eta1p22_mMax6_trkHits10_v3',
    'HLT_DoubleEle4_eta1p22_mMax6_v4',
    'HLT_DoubleEle4p5_eta1p22_mMax6_dz0p8_v3',
    'HLT_DoubleEle4p5_eta1p22_mMax6_trkHits10_v3',
    'HLT_DoubleEle4p5_eta1p22_mMax6_v4',
    'HLT_DoubleEle5_eta1p22_mMax6_dz0p8_v3',
    'HLT_DoubleEle5_eta1p22_mMax6_trkHits10_v3',
    'HLT_DoubleEle5_eta1p22_mMax6_v4',
    'HLT_DoubleEle5p5_eta1p22_mMax6_dz0p8_v3',
    'HLT_DoubleEle5p5_eta1p22_mMax6_trkHits10_v3',
    'HLT_DoubleEle5p5_eta1p22_mMax6_v4',
    'HLT_DoubleEle6_eta1p22_mMax6_dz0p8_v3',
    'HLT_DoubleEle6_eta1p22_mMax6_trkHits10_v3',
    'HLT_DoubleEle6_eta1p22_mMax6_v4',
    'HLT_DoubleEle6p5_eta1p22_mMax6_dz0p8_v3',
    'HLT_DoubleEle6p5_eta1p22_mMax6_trkHits10_v3',
    'HLT_DoubleEle6p5_eta1p22_mMax6_v4',
    'HLT_DoubleEle7_eta1p22_mMax6_dz0p8_v3',
    'HLT_DoubleEle7_eta1p22_mMax6_trkHits10_v3',
    'HLT_DoubleEle7_eta1p22_mMax6_v4',
    'HLT_DoubleEle7p5_eta1p22_mMax6_dz0p8_v3',
    'HLT_DoubleEle7p5_eta1p22_mMax6_trkHits10_v3',
    'HLT_DoubleEle7p5_eta1p22_mMax6_v4',
    'HLT_DoubleEle8_eta1p22_mMax6_dz0p8_v3',
    'HLT_DoubleEle8_eta1p22_mMax6_trkHits10_v3',
    'HLT_DoubleEle8_eta1p22_mMax6_v4',
    'HLT_DoubleEle8p5_eta1p22_mMax6_dz0p8_v3',
    'HLT_DoubleEle8p5_eta1p22_mMax6_trkHits10_v3',
    'HLT_DoubleEle8p5_eta1p22_mMax6_v4',
    'HLT_DoubleEle9_eta1p22_mMax6_dz0p8_v3',
    'HLT_DoubleEle9_eta1p22_mMax6_trkHits10_v3',
    'HLT_DoubleEle9_eta1p22_mMax6_v4',
    'HLT_DoubleEle9p5_eta1p22_mMax6_dz0p8_v3',
    'HLT_DoubleEle9p5_eta1p22_mMax6_trkHits10_v3',
    'HLT_DoubleEle9p5_eta1p22_mMax6_v4',
    'HLT_SingleEle8_SingleEGL1_v3',
    'HLT_SingleEle8_v3'
)


# stream ParkingDoubleMuonLowMass0

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingDoubleMuonLowMass0_datasetParkingDoubleMuonLowMass0_selector
streamParkingDoubleMuonLowMass0_datasetParkingDoubleMuonLowMass0_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingDoubleMuonLowMass0_datasetParkingDoubleMuonLowMass0_selector.l1tResults = cms.InputTag('')
streamParkingDoubleMuonLowMass0_datasetParkingDoubleMuonLowMass0_selector.throw      = cms.bool(False)
streamParkingDoubleMuonLowMass0_datasetParkingDoubleMuonLowMass0_selector.triggerConditions = cms.vstring(
    'HLT_Dimuon0_Jpsi3p5_Muon2_v9',
    'HLT_Dimuon0_Jpsi_L1_4R_0er1p5R_v11',
    'HLT_Dimuon0_Jpsi_L1_NoOS_v11',
    'HLT_Dimuon0_Jpsi_NoVertexing_L1_4R_0er1p5R_v11',
    'HLT_Dimuon0_Jpsi_NoVertexing_NoOS_v11',
    'HLT_Dimuon0_Jpsi_NoVertexing_v12',
    'HLT_Dimuon0_Jpsi_v12',
    'HLT_Dimuon0_LowMass_L1_0er1p5R_v11',
    'HLT_Dimuon0_LowMass_L1_0er1p5_v12',
    'HLT_Dimuon0_LowMass_L1_4R_v11',
    'HLT_Dimuon0_LowMass_L1_4_v12',
    'HLT_Dimuon0_LowMass_L1_TM530_v10',
    'HLT_Dimuon0_LowMass_v12',
    'HLT_Dimuon0_Upsilon_L1_4p5_v13',
    'HLT_Dimuon0_Upsilon_L1_4p5er2p0M_v11',
    'HLT_Dimuon0_Upsilon_L1_4p5er2p0_v13',
    'HLT_Dimuon0_Upsilon_Muon_NoL1Mass_v10',
    'HLT_Dimuon0_Upsilon_NoVertexing_v11',
    'HLT_Dimuon10_Upsilon_y1p4_v5',
    'HLT_Dimuon12_Upsilon_y1p4_v6',
    'HLT_Dimuon14_Phi_Barrel_Seagulls_v11',
    'HLT_Dimuon14_PsiPrime_noCorrL1_v9',
    'HLT_Dimuon14_PsiPrime_v17',
    'HLT_Dimuon18_PsiPrime_noCorrL1_v10',
    'HLT_Dimuon18_PsiPrime_v18',
    'HLT_Dimuon24_Phi_noCorrL1_v10',
    'HLT_Dimuon24_Upsilon_noCorrL1_v10',
    'HLT_Dimuon25_Jpsi_noCorrL1_v10',
    'HLT_Dimuon25_Jpsi_v18',
    'HLT_DoubleMu2_Jpsi_DoubleTrk1_Phi1p05_v10',
    'HLT_DoubleMu3_DoubleEle7p5_CaloIdL_TrackIdL_Upsilon_v8',
    'HLT_DoubleMu3_TkMu_DsTau3Mu_v8',
    'HLT_DoubleMu3_Trk_Tau3mu_NoL1Mass_v10',
    'HLT_DoubleMu3_Trk_Tau3mu_v16',
    'HLT_DoubleMu4_3_Bs_v19',
    'HLT_DoubleMu4_3_Displaced_Photon4_BsToMMG_v4',
    'HLT_DoubleMu4_3_Jpsi_v19',
    'HLT_DoubleMu4_3_LowMass_v5',
    'HLT_DoubleMu4_3_Photon4_BsToMMG_v4',
    'HLT_DoubleMu4_JpsiTrkTrk_Displaced_v11',
    'HLT_DoubleMu4_JpsiTrk_Bc_v4',
    'HLT_DoubleMu4_Jpsi_Displaced_v11',
    'HLT_DoubleMu4_Jpsi_NoVertexing_v11',
    'HLT_DoubleMu4_LowMass_Displaced_v5',
    'HLT_DoubleMu4_MuMuTrk_Displaced_v19',
    'HLT_DoubleMu5_Upsilon_DoubleEle3_CaloIdL_TrackIdL_v8',
    'HLT_Mu25_TkMu0_Phi_v12',
    'HLT_Mu30_TkMu0_Psi_v5',
    'HLT_Mu30_TkMu0_Upsilon_v5',
    'HLT_Mu4_L1DoubleMu_v5',
    'HLT_Mu7p5_L2Mu2_Jpsi_v14',
    'HLT_Mu7p5_L2Mu2_Upsilon_v14',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_Charge1_v8',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_v8',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15_Charge1_v8',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15_v8',
    'HLT_Trimuon5_3p5_2_Upsilon_Muon_v9',
    'HLT_TrimuonOpen_5_3p5_2_Upsilon_Muon_v7'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingDoubleMuonLowMass0_datasetParkingDoubleMuonLowMass1_selector
streamParkingDoubleMuonLowMass0_datasetParkingDoubleMuonLowMass1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingDoubleMuonLowMass0_datasetParkingDoubleMuonLowMass1_selector.l1tResults = cms.InputTag('')
streamParkingDoubleMuonLowMass0_datasetParkingDoubleMuonLowMass1_selector.throw      = cms.bool(False)
streamParkingDoubleMuonLowMass0_datasetParkingDoubleMuonLowMass1_selector.triggerConditions = cms.vstring(
    'HLT_Dimuon0_Jpsi3p5_Muon2_v9',
    'HLT_Dimuon0_Jpsi_L1_4R_0er1p5R_v11',
    'HLT_Dimuon0_Jpsi_L1_NoOS_v11',
    'HLT_Dimuon0_Jpsi_NoVertexing_L1_4R_0er1p5R_v11',
    'HLT_Dimuon0_Jpsi_NoVertexing_NoOS_v11',
    'HLT_Dimuon0_Jpsi_NoVertexing_v12',
    'HLT_Dimuon0_Jpsi_v12',
    'HLT_Dimuon0_LowMass_L1_0er1p5R_v11',
    'HLT_Dimuon0_LowMass_L1_0er1p5_v12',
    'HLT_Dimuon0_LowMass_L1_4R_v11',
    'HLT_Dimuon0_LowMass_L1_4_v12',
    'HLT_Dimuon0_LowMass_L1_TM530_v10',
    'HLT_Dimuon0_LowMass_v12',
    'HLT_Dimuon0_Upsilon_L1_4p5_v13',
    'HLT_Dimuon0_Upsilon_L1_4p5er2p0M_v11',
    'HLT_Dimuon0_Upsilon_L1_4p5er2p0_v13',
    'HLT_Dimuon0_Upsilon_Muon_NoL1Mass_v10',
    'HLT_Dimuon0_Upsilon_NoVertexing_v11',
    'HLT_Dimuon10_Upsilon_y1p4_v5',
    'HLT_Dimuon12_Upsilon_y1p4_v6',
    'HLT_Dimuon14_Phi_Barrel_Seagulls_v11',
    'HLT_Dimuon14_PsiPrime_noCorrL1_v9',
    'HLT_Dimuon14_PsiPrime_v17',
    'HLT_Dimuon18_PsiPrime_noCorrL1_v10',
    'HLT_Dimuon18_PsiPrime_v18',
    'HLT_Dimuon24_Phi_noCorrL1_v10',
    'HLT_Dimuon24_Upsilon_noCorrL1_v10',
    'HLT_Dimuon25_Jpsi_noCorrL1_v10',
    'HLT_Dimuon25_Jpsi_v18',
    'HLT_DoubleMu2_Jpsi_DoubleTrk1_Phi1p05_v10',
    'HLT_DoubleMu3_DoubleEle7p5_CaloIdL_TrackIdL_Upsilon_v8',
    'HLT_DoubleMu3_TkMu_DsTau3Mu_v8',
    'HLT_DoubleMu3_Trk_Tau3mu_NoL1Mass_v10',
    'HLT_DoubleMu3_Trk_Tau3mu_v16',
    'HLT_DoubleMu4_3_Bs_v19',
    'HLT_DoubleMu4_3_Displaced_Photon4_BsToMMG_v4',
    'HLT_DoubleMu4_3_Jpsi_v19',
    'HLT_DoubleMu4_3_LowMass_v5',
    'HLT_DoubleMu4_3_Photon4_BsToMMG_v4',
    'HLT_DoubleMu4_JpsiTrkTrk_Displaced_v11',
    'HLT_DoubleMu4_JpsiTrk_Bc_v4',
    'HLT_DoubleMu4_Jpsi_Displaced_v11',
    'HLT_DoubleMu4_Jpsi_NoVertexing_v11',
    'HLT_DoubleMu4_LowMass_Displaced_v5',
    'HLT_DoubleMu4_MuMuTrk_Displaced_v19',
    'HLT_DoubleMu5_Upsilon_DoubleEle3_CaloIdL_TrackIdL_v8',
    'HLT_Mu25_TkMu0_Phi_v12',
    'HLT_Mu30_TkMu0_Psi_v5',
    'HLT_Mu30_TkMu0_Upsilon_v5',
    'HLT_Mu4_L1DoubleMu_v5',
    'HLT_Mu7p5_L2Mu2_Jpsi_v14',
    'HLT_Mu7p5_L2Mu2_Upsilon_v14',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_Charge1_v8',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_v8',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15_Charge1_v8',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15_v8',
    'HLT_Trimuon5_3p5_2_Upsilon_Muon_v9',
    'HLT_TrimuonOpen_5_3p5_2_Upsilon_Muon_v7'
)


# stream ParkingDoubleMuonLowMass1

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingDoubleMuonLowMass1_datasetParkingDoubleMuonLowMass2_selector
streamParkingDoubleMuonLowMass1_datasetParkingDoubleMuonLowMass2_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingDoubleMuonLowMass1_datasetParkingDoubleMuonLowMass2_selector.l1tResults = cms.InputTag('')
streamParkingDoubleMuonLowMass1_datasetParkingDoubleMuonLowMass2_selector.throw      = cms.bool(False)
streamParkingDoubleMuonLowMass1_datasetParkingDoubleMuonLowMass2_selector.triggerConditions = cms.vstring(
    'HLT_Dimuon0_Jpsi3p5_Muon2_v9',
    'HLT_Dimuon0_Jpsi_L1_4R_0er1p5R_v11',
    'HLT_Dimuon0_Jpsi_L1_NoOS_v11',
    'HLT_Dimuon0_Jpsi_NoVertexing_L1_4R_0er1p5R_v11',
    'HLT_Dimuon0_Jpsi_NoVertexing_NoOS_v11',
    'HLT_Dimuon0_Jpsi_NoVertexing_v12',
    'HLT_Dimuon0_Jpsi_v12',
    'HLT_Dimuon0_LowMass_L1_0er1p5R_v11',
    'HLT_Dimuon0_LowMass_L1_0er1p5_v12',
    'HLT_Dimuon0_LowMass_L1_4R_v11',
    'HLT_Dimuon0_LowMass_L1_4_v12',
    'HLT_Dimuon0_LowMass_L1_TM530_v10',
    'HLT_Dimuon0_LowMass_v12',
    'HLT_Dimuon0_Upsilon_L1_4p5_v13',
    'HLT_Dimuon0_Upsilon_L1_4p5er2p0M_v11',
    'HLT_Dimuon0_Upsilon_L1_4p5er2p0_v13',
    'HLT_Dimuon0_Upsilon_Muon_NoL1Mass_v10',
    'HLT_Dimuon0_Upsilon_NoVertexing_v11',
    'HLT_Dimuon10_Upsilon_y1p4_v5',
    'HLT_Dimuon12_Upsilon_y1p4_v6',
    'HLT_Dimuon14_Phi_Barrel_Seagulls_v11',
    'HLT_Dimuon14_PsiPrime_noCorrL1_v9',
    'HLT_Dimuon14_PsiPrime_v17',
    'HLT_Dimuon18_PsiPrime_noCorrL1_v10',
    'HLT_Dimuon18_PsiPrime_v18',
    'HLT_Dimuon24_Phi_noCorrL1_v10',
    'HLT_Dimuon24_Upsilon_noCorrL1_v10',
    'HLT_Dimuon25_Jpsi_noCorrL1_v10',
    'HLT_Dimuon25_Jpsi_v18',
    'HLT_DoubleMu2_Jpsi_DoubleTrk1_Phi1p05_v10',
    'HLT_DoubleMu3_DoubleEle7p5_CaloIdL_TrackIdL_Upsilon_v8',
    'HLT_DoubleMu3_TkMu_DsTau3Mu_v8',
    'HLT_DoubleMu3_Trk_Tau3mu_NoL1Mass_v10',
    'HLT_DoubleMu3_Trk_Tau3mu_v16',
    'HLT_DoubleMu4_3_Bs_v19',
    'HLT_DoubleMu4_3_Displaced_Photon4_BsToMMG_v4',
    'HLT_DoubleMu4_3_Jpsi_v19',
    'HLT_DoubleMu4_3_LowMass_v5',
    'HLT_DoubleMu4_3_Photon4_BsToMMG_v4',
    'HLT_DoubleMu4_JpsiTrkTrk_Displaced_v11',
    'HLT_DoubleMu4_JpsiTrk_Bc_v4',
    'HLT_DoubleMu4_Jpsi_Displaced_v11',
    'HLT_DoubleMu4_Jpsi_NoVertexing_v11',
    'HLT_DoubleMu4_LowMass_Displaced_v5',
    'HLT_DoubleMu4_MuMuTrk_Displaced_v19',
    'HLT_DoubleMu5_Upsilon_DoubleEle3_CaloIdL_TrackIdL_v8',
    'HLT_Mu25_TkMu0_Phi_v12',
    'HLT_Mu30_TkMu0_Psi_v5',
    'HLT_Mu30_TkMu0_Upsilon_v5',
    'HLT_Mu4_L1DoubleMu_v5',
    'HLT_Mu7p5_L2Mu2_Jpsi_v14',
    'HLT_Mu7p5_L2Mu2_Upsilon_v14',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_Charge1_v8',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_v8',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15_Charge1_v8',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15_v8',
    'HLT_Trimuon5_3p5_2_Upsilon_Muon_v9',
    'HLT_TrimuonOpen_5_3p5_2_Upsilon_Muon_v7'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingDoubleMuonLowMass1_datasetParkingDoubleMuonLowMass3_selector
streamParkingDoubleMuonLowMass1_datasetParkingDoubleMuonLowMass3_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingDoubleMuonLowMass1_datasetParkingDoubleMuonLowMass3_selector.l1tResults = cms.InputTag('')
streamParkingDoubleMuonLowMass1_datasetParkingDoubleMuonLowMass3_selector.throw      = cms.bool(False)
streamParkingDoubleMuonLowMass1_datasetParkingDoubleMuonLowMass3_selector.triggerConditions = cms.vstring(
    'HLT_Dimuon0_Jpsi3p5_Muon2_v9',
    'HLT_Dimuon0_Jpsi_L1_4R_0er1p5R_v11',
    'HLT_Dimuon0_Jpsi_L1_NoOS_v11',
    'HLT_Dimuon0_Jpsi_NoVertexing_L1_4R_0er1p5R_v11',
    'HLT_Dimuon0_Jpsi_NoVertexing_NoOS_v11',
    'HLT_Dimuon0_Jpsi_NoVertexing_v12',
    'HLT_Dimuon0_Jpsi_v12',
    'HLT_Dimuon0_LowMass_L1_0er1p5R_v11',
    'HLT_Dimuon0_LowMass_L1_0er1p5_v12',
    'HLT_Dimuon0_LowMass_L1_4R_v11',
    'HLT_Dimuon0_LowMass_L1_4_v12',
    'HLT_Dimuon0_LowMass_L1_TM530_v10',
    'HLT_Dimuon0_LowMass_v12',
    'HLT_Dimuon0_Upsilon_L1_4p5_v13',
    'HLT_Dimuon0_Upsilon_L1_4p5er2p0M_v11',
    'HLT_Dimuon0_Upsilon_L1_4p5er2p0_v13',
    'HLT_Dimuon0_Upsilon_Muon_NoL1Mass_v10',
    'HLT_Dimuon0_Upsilon_NoVertexing_v11',
    'HLT_Dimuon10_Upsilon_y1p4_v5',
    'HLT_Dimuon12_Upsilon_y1p4_v6',
    'HLT_Dimuon14_Phi_Barrel_Seagulls_v11',
    'HLT_Dimuon14_PsiPrime_noCorrL1_v9',
    'HLT_Dimuon14_PsiPrime_v17',
    'HLT_Dimuon18_PsiPrime_noCorrL1_v10',
    'HLT_Dimuon18_PsiPrime_v18',
    'HLT_Dimuon24_Phi_noCorrL1_v10',
    'HLT_Dimuon24_Upsilon_noCorrL1_v10',
    'HLT_Dimuon25_Jpsi_noCorrL1_v10',
    'HLT_Dimuon25_Jpsi_v18',
    'HLT_DoubleMu2_Jpsi_DoubleTrk1_Phi1p05_v10',
    'HLT_DoubleMu3_DoubleEle7p5_CaloIdL_TrackIdL_Upsilon_v8',
    'HLT_DoubleMu3_TkMu_DsTau3Mu_v8',
    'HLT_DoubleMu3_Trk_Tau3mu_NoL1Mass_v10',
    'HLT_DoubleMu3_Trk_Tau3mu_v16',
    'HLT_DoubleMu4_3_Bs_v19',
    'HLT_DoubleMu4_3_Displaced_Photon4_BsToMMG_v4',
    'HLT_DoubleMu4_3_Jpsi_v19',
    'HLT_DoubleMu4_3_LowMass_v5',
    'HLT_DoubleMu4_3_Photon4_BsToMMG_v4',
    'HLT_DoubleMu4_JpsiTrkTrk_Displaced_v11',
    'HLT_DoubleMu4_JpsiTrk_Bc_v4',
    'HLT_DoubleMu4_Jpsi_Displaced_v11',
    'HLT_DoubleMu4_Jpsi_NoVertexing_v11',
    'HLT_DoubleMu4_LowMass_Displaced_v5',
    'HLT_DoubleMu4_MuMuTrk_Displaced_v19',
    'HLT_DoubleMu5_Upsilon_DoubleEle3_CaloIdL_TrackIdL_v8',
    'HLT_Mu25_TkMu0_Phi_v12',
    'HLT_Mu30_TkMu0_Psi_v5',
    'HLT_Mu30_TkMu0_Upsilon_v5',
    'HLT_Mu4_L1DoubleMu_v5',
    'HLT_Mu7p5_L2Mu2_Jpsi_v14',
    'HLT_Mu7p5_L2Mu2_Upsilon_v14',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_Charge1_v8',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_v8',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15_Charge1_v8',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15_v8',
    'HLT_Trimuon5_3p5_2_Upsilon_Muon_v9',
    'HLT_TrimuonOpen_5_3p5_2_Upsilon_Muon_v7'
)


# stream ParkingDoubleMuonLowMass2

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingDoubleMuonLowMass2_datasetParkingDoubleMuonLowMass4_selector
streamParkingDoubleMuonLowMass2_datasetParkingDoubleMuonLowMass4_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingDoubleMuonLowMass2_datasetParkingDoubleMuonLowMass4_selector.l1tResults = cms.InputTag('')
streamParkingDoubleMuonLowMass2_datasetParkingDoubleMuonLowMass4_selector.throw      = cms.bool(False)
streamParkingDoubleMuonLowMass2_datasetParkingDoubleMuonLowMass4_selector.triggerConditions = cms.vstring(
    'HLT_Dimuon0_Jpsi3p5_Muon2_v9',
    'HLT_Dimuon0_Jpsi_L1_4R_0er1p5R_v11',
    'HLT_Dimuon0_Jpsi_L1_NoOS_v11',
    'HLT_Dimuon0_Jpsi_NoVertexing_L1_4R_0er1p5R_v11',
    'HLT_Dimuon0_Jpsi_NoVertexing_NoOS_v11',
    'HLT_Dimuon0_Jpsi_NoVertexing_v12',
    'HLT_Dimuon0_Jpsi_v12',
    'HLT_Dimuon0_LowMass_L1_0er1p5R_v11',
    'HLT_Dimuon0_LowMass_L1_0er1p5_v12',
    'HLT_Dimuon0_LowMass_L1_4R_v11',
    'HLT_Dimuon0_LowMass_L1_4_v12',
    'HLT_Dimuon0_LowMass_L1_TM530_v10',
    'HLT_Dimuon0_LowMass_v12',
    'HLT_Dimuon0_Upsilon_L1_4p5_v13',
    'HLT_Dimuon0_Upsilon_L1_4p5er2p0M_v11',
    'HLT_Dimuon0_Upsilon_L1_4p5er2p0_v13',
    'HLT_Dimuon0_Upsilon_Muon_NoL1Mass_v10',
    'HLT_Dimuon0_Upsilon_NoVertexing_v11',
    'HLT_Dimuon10_Upsilon_y1p4_v5',
    'HLT_Dimuon12_Upsilon_y1p4_v6',
    'HLT_Dimuon14_Phi_Barrel_Seagulls_v11',
    'HLT_Dimuon14_PsiPrime_noCorrL1_v9',
    'HLT_Dimuon14_PsiPrime_v17',
    'HLT_Dimuon18_PsiPrime_noCorrL1_v10',
    'HLT_Dimuon18_PsiPrime_v18',
    'HLT_Dimuon24_Phi_noCorrL1_v10',
    'HLT_Dimuon24_Upsilon_noCorrL1_v10',
    'HLT_Dimuon25_Jpsi_noCorrL1_v10',
    'HLT_Dimuon25_Jpsi_v18',
    'HLT_DoubleMu2_Jpsi_DoubleTrk1_Phi1p05_v10',
    'HLT_DoubleMu3_DoubleEle7p5_CaloIdL_TrackIdL_Upsilon_v8',
    'HLT_DoubleMu3_TkMu_DsTau3Mu_v8',
    'HLT_DoubleMu3_Trk_Tau3mu_NoL1Mass_v10',
    'HLT_DoubleMu3_Trk_Tau3mu_v16',
    'HLT_DoubleMu4_3_Bs_v19',
    'HLT_DoubleMu4_3_Displaced_Photon4_BsToMMG_v4',
    'HLT_DoubleMu4_3_Jpsi_v19',
    'HLT_DoubleMu4_3_LowMass_v5',
    'HLT_DoubleMu4_3_Photon4_BsToMMG_v4',
    'HLT_DoubleMu4_JpsiTrkTrk_Displaced_v11',
    'HLT_DoubleMu4_JpsiTrk_Bc_v4',
    'HLT_DoubleMu4_Jpsi_Displaced_v11',
    'HLT_DoubleMu4_Jpsi_NoVertexing_v11',
    'HLT_DoubleMu4_LowMass_Displaced_v5',
    'HLT_DoubleMu4_MuMuTrk_Displaced_v19',
    'HLT_DoubleMu5_Upsilon_DoubleEle3_CaloIdL_TrackIdL_v8',
    'HLT_Mu25_TkMu0_Phi_v12',
    'HLT_Mu30_TkMu0_Psi_v5',
    'HLT_Mu30_TkMu0_Upsilon_v5',
    'HLT_Mu4_L1DoubleMu_v5',
    'HLT_Mu7p5_L2Mu2_Jpsi_v14',
    'HLT_Mu7p5_L2Mu2_Upsilon_v14',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_Charge1_v8',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_v8',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15_Charge1_v8',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15_v8',
    'HLT_Trimuon5_3p5_2_Upsilon_Muon_v9',
    'HLT_TrimuonOpen_5_3p5_2_Upsilon_Muon_v7'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingDoubleMuonLowMass2_datasetParkingDoubleMuonLowMass5_selector
streamParkingDoubleMuonLowMass2_datasetParkingDoubleMuonLowMass5_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingDoubleMuonLowMass2_datasetParkingDoubleMuonLowMass5_selector.l1tResults = cms.InputTag('')
streamParkingDoubleMuonLowMass2_datasetParkingDoubleMuonLowMass5_selector.throw      = cms.bool(False)
streamParkingDoubleMuonLowMass2_datasetParkingDoubleMuonLowMass5_selector.triggerConditions = cms.vstring(
    'HLT_Dimuon0_Jpsi3p5_Muon2_v9',
    'HLT_Dimuon0_Jpsi_L1_4R_0er1p5R_v11',
    'HLT_Dimuon0_Jpsi_L1_NoOS_v11',
    'HLT_Dimuon0_Jpsi_NoVertexing_L1_4R_0er1p5R_v11',
    'HLT_Dimuon0_Jpsi_NoVertexing_NoOS_v11',
    'HLT_Dimuon0_Jpsi_NoVertexing_v12',
    'HLT_Dimuon0_Jpsi_v12',
    'HLT_Dimuon0_LowMass_L1_0er1p5R_v11',
    'HLT_Dimuon0_LowMass_L1_0er1p5_v12',
    'HLT_Dimuon0_LowMass_L1_4R_v11',
    'HLT_Dimuon0_LowMass_L1_4_v12',
    'HLT_Dimuon0_LowMass_L1_TM530_v10',
    'HLT_Dimuon0_LowMass_v12',
    'HLT_Dimuon0_Upsilon_L1_4p5_v13',
    'HLT_Dimuon0_Upsilon_L1_4p5er2p0M_v11',
    'HLT_Dimuon0_Upsilon_L1_4p5er2p0_v13',
    'HLT_Dimuon0_Upsilon_Muon_NoL1Mass_v10',
    'HLT_Dimuon0_Upsilon_NoVertexing_v11',
    'HLT_Dimuon10_Upsilon_y1p4_v5',
    'HLT_Dimuon12_Upsilon_y1p4_v6',
    'HLT_Dimuon14_Phi_Barrel_Seagulls_v11',
    'HLT_Dimuon14_PsiPrime_noCorrL1_v9',
    'HLT_Dimuon14_PsiPrime_v17',
    'HLT_Dimuon18_PsiPrime_noCorrL1_v10',
    'HLT_Dimuon18_PsiPrime_v18',
    'HLT_Dimuon24_Phi_noCorrL1_v10',
    'HLT_Dimuon24_Upsilon_noCorrL1_v10',
    'HLT_Dimuon25_Jpsi_noCorrL1_v10',
    'HLT_Dimuon25_Jpsi_v18',
    'HLT_DoubleMu2_Jpsi_DoubleTrk1_Phi1p05_v10',
    'HLT_DoubleMu3_DoubleEle7p5_CaloIdL_TrackIdL_Upsilon_v8',
    'HLT_DoubleMu3_TkMu_DsTau3Mu_v8',
    'HLT_DoubleMu3_Trk_Tau3mu_NoL1Mass_v10',
    'HLT_DoubleMu3_Trk_Tau3mu_v16',
    'HLT_DoubleMu4_3_Bs_v19',
    'HLT_DoubleMu4_3_Displaced_Photon4_BsToMMG_v4',
    'HLT_DoubleMu4_3_Jpsi_v19',
    'HLT_DoubleMu4_3_LowMass_v5',
    'HLT_DoubleMu4_3_Photon4_BsToMMG_v4',
    'HLT_DoubleMu4_JpsiTrkTrk_Displaced_v11',
    'HLT_DoubleMu4_JpsiTrk_Bc_v4',
    'HLT_DoubleMu4_Jpsi_Displaced_v11',
    'HLT_DoubleMu4_Jpsi_NoVertexing_v11',
    'HLT_DoubleMu4_LowMass_Displaced_v5',
    'HLT_DoubleMu4_MuMuTrk_Displaced_v19',
    'HLT_DoubleMu5_Upsilon_DoubleEle3_CaloIdL_TrackIdL_v8',
    'HLT_Mu25_TkMu0_Phi_v12',
    'HLT_Mu30_TkMu0_Psi_v5',
    'HLT_Mu30_TkMu0_Upsilon_v5',
    'HLT_Mu4_L1DoubleMu_v5',
    'HLT_Mu7p5_L2Mu2_Jpsi_v14',
    'HLT_Mu7p5_L2Mu2_Upsilon_v14',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_Charge1_v8',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_v8',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15_Charge1_v8',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15_v8',
    'HLT_Trimuon5_3p5_2_Upsilon_Muon_v9',
    'HLT_TrimuonOpen_5_3p5_2_Upsilon_Muon_v7'
)


# stream ParkingDoubleMuonLowMass3

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingDoubleMuonLowMass3_datasetParkingDoubleMuonLowMass6_selector
streamParkingDoubleMuonLowMass3_datasetParkingDoubleMuonLowMass6_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingDoubleMuonLowMass3_datasetParkingDoubleMuonLowMass6_selector.l1tResults = cms.InputTag('')
streamParkingDoubleMuonLowMass3_datasetParkingDoubleMuonLowMass6_selector.throw      = cms.bool(False)
streamParkingDoubleMuonLowMass3_datasetParkingDoubleMuonLowMass6_selector.triggerConditions = cms.vstring(
    'HLT_Dimuon0_Jpsi3p5_Muon2_v9',
    'HLT_Dimuon0_Jpsi_L1_4R_0er1p5R_v11',
    'HLT_Dimuon0_Jpsi_L1_NoOS_v11',
    'HLT_Dimuon0_Jpsi_NoVertexing_L1_4R_0er1p5R_v11',
    'HLT_Dimuon0_Jpsi_NoVertexing_NoOS_v11',
    'HLT_Dimuon0_Jpsi_NoVertexing_v12',
    'HLT_Dimuon0_Jpsi_v12',
    'HLT_Dimuon0_LowMass_L1_0er1p5R_v11',
    'HLT_Dimuon0_LowMass_L1_0er1p5_v12',
    'HLT_Dimuon0_LowMass_L1_4R_v11',
    'HLT_Dimuon0_LowMass_L1_4_v12',
    'HLT_Dimuon0_LowMass_L1_TM530_v10',
    'HLT_Dimuon0_LowMass_v12',
    'HLT_Dimuon0_Upsilon_L1_4p5_v13',
    'HLT_Dimuon0_Upsilon_L1_4p5er2p0M_v11',
    'HLT_Dimuon0_Upsilon_L1_4p5er2p0_v13',
    'HLT_Dimuon0_Upsilon_Muon_NoL1Mass_v10',
    'HLT_Dimuon0_Upsilon_NoVertexing_v11',
    'HLT_Dimuon10_Upsilon_y1p4_v5',
    'HLT_Dimuon12_Upsilon_y1p4_v6',
    'HLT_Dimuon14_Phi_Barrel_Seagulls_v11',
    'HLT_Dimuon14_PsiPrime_noCorrL1_v9',
    'HLT_Dimuon14_PsiPrime_v17',
    'HLT_Dimuon18_PsiPrime_noCorrL1_v10',
    'HLT_Dimuon18_PsiPrime_v18',
    'HLT_Dimuon24_Phi_noCorrL1_v10',
    'HLT_Dimuon24_Upsilon_noCorrL1_v10',
    'HLT_Dimuon25_Jpsi_noCorrL1_v10',
    'HLT_Dimuon25_Jpsi_v18',
    'HLT_DoubleMu2_Jpsi_DoubleTrk1_Phi1p05_v10',
    'HLT_DoubleMu3_DoubleEle7p5_CaloIdL_TrackIdL_Upsilon_v8',
    'HLT_DoubleMu3_TkMu_DsTau3Mu_v8',
    'HLT_DoubleMu3_Trk_Tau3mu_NoL1Mass_v10',
    'HLT_DoubleMu3_Trk_Tau3mu_v16',
    'HLT_DoubleMu4_3_Bs_v19',
    'HLT_DoubleMu4_3_Displaced_Photon4_BsToMMG_v4',
    'HLT_DoubleMu4_3_Jpsi_v19',
    'HLT_DoubleMu4_3_LowMass_v5',
    'HLT_DoubleMu4_3_Photon4_BsToMMG_v4',
    'HLT_DoubleMu4_JpsiTrkTrk_Displaced_v11',
    'HLT_DoubleMu4_JpsiTrk_Bc_v4',
    'HLT_DoubleMu4_Jpsi_Displaced_v11',
    'HLT_DoubleMu4_Jpsi_NoVertexing_v11',
    'HLT_DoubleMu4_LowMass_Displaced_v5',
    'HLT_DoubleMu4_MuMuTrk_Displaced_v19',
    'HLT_DoubleMu5_Upsilon_DoubleEle3_CaloIdL_TrackIdL_v8',
    'HLT_Mu25_TkMu0_Phi_v12',
    'HLT_Mu30_TkMu0_Psi_v5',
    'HLT_Mu30_TkMu0_Upsilon_v5',
    'HLT_Mu4_L1DoubleMu_v5',
    'HLT_Mu7p5_L2Mu2_Jpsi_v14',
    'HLT_Mu7p5_L2Mu2_Upsilon_v14',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_Charge1_v8',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_v8',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15_Charge1_v8',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15_v8',
    'HLT_Trimuon5_3p5_2_Upsilon_Muon_v9',
    'HLT_TrimuonOpen_5_3p5_2_Upsilon_Muon_v7'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingDoubleMuonLowMass3_datasetParkingDoubleMuonLowMass7_selector
streamParkingDoubleMuonLowMass3_datasetParkingDoubleMuonLowMass7_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingDoubleMuonLowMass3_datasetParkingDoubleMuonLowMass7_selector.l1tResults = cms.InputTag('')
streamParkingDoubleMuonLowMass3_datasetParkingDoubleMuonLowMass7_selector.throw      = cms.bool(False)
streamParkingDoubleMuonLowMass3_datasetParkingDoubleMuonLowMass7_selector.triggerConditions = cms.vstring(
    'HLT_Dimuon0_Jpsi3p5_Muon2_v9',
    'HLT_Dimuon0_Jpsi_L1_4R_0er1p5R_v11',
    'HLT_Dimuon0_Jpsi_L1_NoOS_v11',
    'HLT_Dimuon0_Jpsi_NoVertexing_L1_4R_0er1p5R_v11',
    'HLT_Dimuon0_Jpsi_NoVertexing_NoOS_v11',
    'HLT_Dimuon0_Jpsi_NoVertexing_v12',
    'HLT_Dimuon0_Jpsi_v12',
    'HLT_Dimuon0_LowMass_L1_0er1p5R_v11',
    'HLT_Dimuon0_LowMass_L1_0er1p5_v12',
    'HLT_Dimuon0_LowMass_L1_4R_v11',
    'HLT_Dimuon0_LowMass_L1_4_v12',
    'HLT_Dimuon0_LowMass_L1_TM530_v10',
    'HLT_Dimuon0_LowMass_v12',
    'HLT_Dimuon0_Upsilon_L1_4p5_v13',
    'HLT_Dimuon0_Upsilon_L1_4p5er2p0M_v11',
    'HLT_Dimuon0_Upsilon_L1_4p5er2p0_v13',
    'HLT_Dimuon0_Upsilon_Muon_NoL1Mass_v10',
    'HLT_Dimuon0_Upsilon_NoVertexing_v11',
    'HLT_Dimuon10_Upsilon_y1p4_v5',
    'HLT_Dimuon12_Upsilon_y1p4_v6',
    'HLT_Dimuon14_Phi_Barrel_Seagulls_v11',
    'HLT_Dimuon14_PsiPrime_noCorrL1_v9',
    'HLT_Dimuon14_PsiPrime_v17',
    'HLT_Dimuon18_PsiPrime_noCorrL1_v10',
    'HLT_Dimuon18_PsiPrime_v18',
    'HLT_Dimuon24_Phi_noCorrL1_v10',
    'HLT_Dimuon24_Upsilon_noCorrL1_v10',
    'HLT_Dimuon25_Jpsi_noCorrL1_v10',
    'HLT_Dimuon25_Jpsi_v18',
    'HLT_DoubleMu2_Jpsi_DoubleTrk1_Phi1p05_v10',
    'HLT_DoubleMu3_DoubleEle7p5_CaloIdL_TrackIdL_Upsilon_v8',
    'HLT_DoubleMu3_TkMu_DsTau3Mu_v8',
    'HLT_DoubleMu3_Trk_Tau3mu_NoL1Mass_v10',
    'HLT_DoubleMu3_Trk_Tau3mu_v16',
    'HLT_DoubleMu4_3_Bs_v19',
    'HLT_DoubleMu4_3_Displaced_Photon4_BsToMMG_v4',
    'HLT_DoubleMu4_3_Jpsi_v19',
    'HLT_DoubleMu4_3_LowMass_v5',
    'HLT_DoubleMu4_3_Photon4_BsToMMG_v4',
    'HLT_DoubleMu4_JpsiTrkTrk_Displaced_v11',
    'HLT_DoubleMu4_JpsiTrk_Bc_v4',
    'HLT_DoubleMu4_Jpsi_Displaced_v11',
    'HLT_DoubleMu4_Jpsi_NoVertexing_v11',
    'HLT_DoubleMu4_LowMass_Displaced_v5',
    'HLT_DoubleMu4_MuMuTrk_Displaced_v19',
    'HLT_DoubleMu5_Upsilon_DoubleEle3_CaloIdL_TrackIdL_v8',
    'HLT_Mu25_TkMu0_Phi_v12',
    'HLT_Mu30_TkMu0_Psi_v5',
    'HLT_Mu30_TkMu0_Upsilon_v5',
    'HLT_Mu4_L1DoubleMu_v5',
    'HLT_Mu7p5_L2Mu2_Jpsi_v14',
    'HLT_Mu7p5_L2Mu2_Upsilon_v14',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_Charge1_v8',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_v8',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15_Charge1_v8',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15_v8',
    'HLT_Trimuon5_3p5_2_Upsilon_Muon_v9',
    'HLT_TrimuonOpen_5_3p5_2_Upsilon_Muon_v7'
)


# stream ParkingHH

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingHH_datasetParkingHH_selector
streamParkingHH_datasetParkingHH_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingHH_datasetParkingHH_selector.l1tResults = cms.InputTag('')
streamParkingHH_datasetParkingHH_selector.throw      = cms.bool(False)
streamParkingHH_datasetParkingHH_selector.triggerConditions = cms.vstring(
    'HLT_PFHT280_QuadPFJet30_PNet2BTagMean0p55_v1',
    'HLT_PFHT280_QuadPFJet30_PNet2BTagMean0p60_v1',
    'HLT_PFHT280_QuadPFJet30_v1',
    'HLT_PFHT280_QuadPFJet35_PNet2BTagMean0p60_v1',
    'HLT_PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepJet_4p5_v5',
    'HLT_PFHT330PT30_QuadPFJet_75_60_45_40_v13',
    'HLT_PFHT340_QuadPFJet70_50_40_40_PNet2BTagMean0p70_v2',
    'HLT_PFHT400_FivePFJet_100_100_60_30_30_DoublePFBTagDeepJet_4p5_v5',
    'HLT_PFHT400_FivePFJet_100_100_60_30_30_v12',
    'HLT_PFHT400_FivePFJet_120_120_60_30_30_DoublePFBTagDeepJet_4p5_v5',
    'HLT_PFHT400_SixPFJet32_DoublePFBTagDeepJet_2p94_v5',
    'HLT_PFHT400_SixPFJet32_PNet2BTagMean0p50_v1',
    'HLT_PFHT400_SixPFJet32_v13',
    'HLT_PFHT450_SixPFJet36_PFBTagDeepJet_1p59_v5',
    'HLT_PFHT450_SixPFJet36_PNetBTag0p35_v1',
    'HLT_PFHT450_SixPFJet36_v12'
)


# stream ParkingLLP

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingLLP_datasetParkingLLP_selector
streamParkingLLP_datasetParkingLLP_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingLLP_datasetParkingLLP_selector.l1tResults = cms.InputTag('')
streamParkingLLP_datasetParkingLLP_selector.throw      = cms.bool(False)
streamParkingLLP_datasetParkingLLP_selector.triggerConditions = cms.vstring(
    'HLT_HT350_DelayedJet40_SingleDelay1p5To3p5nsInclusive_v1',
    'HLT_HT350_DelayedJet40_SingleDelay1p6To3p5nsInclusive_v1',
    'HLT_HT350_DelayedJet40_SingleDelay1p75To3p5nsInclusive_v1',
    'HLT_HT360_DisplacedDijet40_Inclusive1PtrkShortSig5_v1',
    'HLT_HT360_DisplacedDijet45_Inclusive1PtrkShortSig5_v1',
    'HLT_HT390_DisplacedDijet40_Inclusive1PtrkShortSig5_v1',
    'HLT_HT390_DisplacedDijet45_Inclusive1PtrkShortSig5_v1',
    'HLT_HT390eta2p0_DisplacedDijet40_Inclusive1PtrkShortSig5_v1',
    'HLT_HT430_DelayedJet40_SingleDelay1To1p5nsInclusive_v1',
    'HLT_HT430_DelayedJet40_SingleDelay1p1To1p6nsInclusive_v1',
    'HLT_HT430_DelayedJet40_SingleDelay1p25To1p75nsInclusive_v1',
    'HLT_HT430_DisplacedDijet40_DisplacedTrack_v17',
    'HLT_HT430_DisplacedDijet40_Inclusive1PtrkShortSig5_v5',
    'HLT_HT650_DisplacedDijet60_Inclusive_v17',
    'HLT_L1Tau_DelayedJet40_SingleDelay2p5To4nsInclusive_v1',
    'HLT_L1Tau_DelayedJet40_SingleDelay2p6To4nsInclusive_v1',
    'HLT_L1Tau_DelayedJet40_SingleDelay2p75To4nsInclusive_v1'
)


# stream ParkingVBF0

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingVBF0_datasetParkingVBF0_selector
streamParkingVBF0_datasetParkingVBF0_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingVBF0_datasetParkingVBF0_selector.l1tResults = cms.InputTag('')
streamParkingVBF0_datasetParkingVBF0_selector.throw      = cms.bool(False)
streamParkingVBF0_datasetParkingVBF0_selector.triggerConditions = cms.vstring(
    'HLT_DiJet110_35_Mjj650_PFMET110_v13',
    'HLT_DiJet110_35_Mjj650_PFMET120_v13',
    'HLT_DiJet110_35_Mjj650_PFMET130_v13',
    'HLT_DoublePFJets40_Mass500_MediumDeepTauPFTauHPS45_L2NN_MediumDeepTauPFTauHPS20_eta2p1_v4',
    'HLT_Mu10_TrkIsoVVL_DiPFJet40_DEta3p5_MJJ750_HTT350_PFMETNoMu60_v20',
    'HLT_Mu4_TrkIsoVVL_DiPFJet90_40_DEta3p5_MJJ750_HTT300_PFMETNoMu60_v20',
    'HLT_Mu8_TrkIsoVVL_DiPFJet40_DEta3p5_MJJ750_HTT300_PFMETNoMu60_v21',
    'HLT_QuadPFJet100_88_70_30_PNet1CvsAll0p5_VBF3Tight_v2',
    'HLT_QuadPFJet103_88_75_15_DoublePFBTagDeepJet_1p3_7p7_VBF1_v5',
    'HLT_QuadPFJet103_88_75_15_PFBTagDeepJet_1p3_VBF2_v5',
    'HLT_QuadPFJet105_88_75_30_PNet1CvsAll0p5_VBF3Tight_v2',
    'HLT_QuadPFJet105_88_76_15_DoublePFBTagDeepJet_1p3_7p7_VBF1_v5',
    'HLT_QuadPFJet105_88_76_15_PFBTagDeepJet_1p3_VBF2_v5',
    'HLT_QuadPFJet111_90_80_15_DoublePFBTagDeepJet_1p3_7p7_VBF1_v5',
    'HLT_QuadPFJet111_90_80_15_PFBTagDeepJet_1p3_VBF2_v5',
    'HLT_QuadPFJet111_90_80_30_PNet1CvsAll0p6_VBF3Tight_v2',
    'HLT_TripleJet110_35_35_Mjj650_PFMET110_v13',
    'HLT_TripleJet110_35_35_Mjj650_PFMET120_v13',
    'HLT_TripleJet110_35_35_Mjj650_PFMET130_v13',
    'HLT_VBF_DiPFJet105_40_Mjj1000_Detajj3p5_TriplePFJet_v2',
    'HLT_VBF_DiPFJet105_40_Mjj1000_Detajj3p5_v2',
    'HLT_VBF_DiPFJet110_40_Mjj1000_Detajj3p5_TriplePFJet_v1',
    'HLT_VBF_DiPFJet110_40_Mjj1000_Detajj3p5_v1',
    'HLT_VBF_DiPFJet125_45_Mjj1000_Detajj3p5_TriplePFJet_v1',
    'HLT_VBF_DiPFJet125_45_Mjj1000_Detajj3p5_v1',
    'HLT_VBF_DiPFJet125_45_Mjj720_Detajj3p0_TriplePFJet_v2',
    'HLT_VBF_DiPFJet125_45_Mjj720_Detajj3p0_v2',
    'HLT_VBF_DiPFJet45_Mjj500_Detajj2p5_Ele12_eta2p1_WPTight_Gsf_v1',
    'HLT_VBF_DiPFJet45_Mjj500_Detajj2p5_Ele17_eta2p1_WPTight_Gsf_v1',
    'HLT_VBF_DiPFJet45_Mjj500_Detajj2p5_MediumDeepTauPFTauHPS45_L2NN_eta2p1_v1',
    'HLT_VBF_DiPFJet45_Mjj500_Detajj2p5_Photon12_v1',
    'HLT_VBF_DiPFJet45_Mjj500_Detajj2p5_Photon17_v1',
    'HLT_VBF_DiPFJet50_Mjj500_Detajj2p5_Ele22_eta2p1_WPTight_Gsf_v1',
    'HLT_VBF_DiPFJet50_Mjj500_Detajj2p5_Photon22_v1',
    'HLT_VBF_DiPFJet70_40_Mjj600_Detajj2p5_DiPFJet60_JetMatchingFiveJet_v2',
    'HLT_VBF_DiPFJet70_40_Mjj600_Detajj2p5_DiPFJet60_JetMatchingQuadJet_v2',
    'HLT_VBF_DiPFJet70_40_Mjj600_Detajj2p5_DiPFJet60_JetMatchingSixJet_v2',
    'HLT_VBF_DiPFJet75_40_Mjj500_Detajj2p5_PFMET85_TriplePFJet_v2',
    'HLT_VBF_DiPFJet75_40_Mjj500_Detajj2p5_PFMET85_v2',
    'HLT_VBF_DiPFJet75_45_Mjj600_Detajj2p5_DiPFJet60_JetMatchingFiveJet_v1',
    'HLT_VBF_DiPFJet75_45_Mjj600_Detajj2p5_DiPFJet60_JetMatchingQuadJet_v1',
    'HLT_VBF_DiPFJet75_45_Mjj600_Detajj2p5_DiPFJet60_JetMatchingSixJet_v1',
    'HLT_VBF_DiPFJet80_45_Mjj500_Detajj2p5_PFMET85_TriplePFJet_v1',
    'HLT_VBF_DiPFJet80_45_Mjj500_Detajj2p5_PFMET85_v1',
    'HLT_VBF_DiPFJet90_40_Mjj600_Detajj2p5_Mu3_TrkIsoVVL_TriplePFJet_v2',
    'HLT_VBF_DiPFJet90_40_Mjj600_Detajj2p5_Mu3_TrkIsoVVL_v2',
    'HLT_VBF_DiPFJet95_45_Mjj600_Detajj2p5_Mu3_TrkIsoVVL_TriplePFJet_v1',
    'HLT_VBF_DiPFJet95_45_Mjj600_Detajj2p5_Mu3_TrkIsoVVL_v1',
    'HLT_VBF_DoubleMediumDeepTauPFTauHPS20_eta2p1_v5'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingVBF0_datasetParkingVBF1_selector
streamParkingVBF0_datasetParkingVBF1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingVBF0_datasetParkingVBF1_selector.l1tResults = cms.InputTag('')
streamParkingVBF0_datasetParkingVBF1_selector.throw      = cms.bool(False)
streamParkingVBF0_datasetParkingVBF1_selector.triggerConditions = cms.vstring(
    'HLT_DiJet110_35_Mjj650_PFMET110_v13',
    'HLT_DiJet110_35_Mjj650_PFMET120_v13',
    'HLT_DiJet110_35_Mjj650_PFMET130_v13',
    'HLT_DoublePFJets40_Mass500_MediumDeepTauPFTauHPS45_L2NN_MediumDeepTauPFTauHPS20_eta2p1_v4',
    'HLT_Mu10_TrkIsoVVL_DiPFJet40_DEta3p5_MJJ750_HTT350_PFMETNoMu60_v20',
    'HLT_Mu4_TrkIsoVVL_DiPFJet90_40_DEta3p5_MJJ750_HTT300_PFMETNoMu60_v20',
    'HLT_Mu8_TrkIsoVVL_DiPFJet40_DEta3p5_MJJ750_HTT300_PFMETNoMu60_v21',
    'HLT_QuadPFJet100_88_70_30_PNet1CvsAll0p5_VBF3Tight_v2',
    'HLT_QuadPFJet103_88_75_15_DoublePFBTagDeepJet_1p3_7p7_VBF1_v5',
    'HLT_QuadPFJet103_88_75_15_PFBTagDeepJet_1p3_VBF2_v5',
    'HLT_QuadPFJet105_88_75_30_PNet1CvsAll0p5_VBF3Tight_v2',
    'HLT_QuadPFJet105_88_76_15_DoublePFBTagDeepJet_1p3_7p7_VBF1_v5',
    'HLT_QuadPFJet105_88_76_15_PFBTagDeepJet_1p3_VBF2_v5',
    'HLT_QuadPFJet111_90_80_15_DoublePFBTagDeepJet_1p3_7p7_VBF1_v5',
    'HLT_QuadPFJet111_90_80_15_PFBTagDeepJet_1p3_VBF2_v5',
    'HLT_QuadPFJet111_90_80_30_PNet1CvsAll0p6_VBF3Tight_v2',
    'HLT_TripleJet110_35_35_Mjj650_PFMET110_v13',
    'HLT_TripleJet110_35_35_Mjj650_PFMET120_v13',
    'HLT_TripleJet110_35_35_Mjj650_PFMET130_v13',
    'HLT_VBF_DiPFJet105_40_Mjj1000_Detajj3p5_TriplePFJet_v2',
    'HLT_VBF_DiPFJet105_40_Mjj1000_Detajj3p5_v2',
    'HLT_VBF_DiPFJet110_40_Mjj1000_Detajj3p5_TriplePFJet_v1',
    'HLT_VBF_DiPFJet110_40_Mjj1000_Detajj3p5_v1',
    'HLT_VBF_DiPFJet125_45_Mjj1000_Detajj3p5_TriplePFJet_v1',
    'HLT_VBF_DiPFJet125_45_Mjj1000_Detajj3p5_v1',
    'HLT_VBF_DiPFJet125_45_Mjj720_Detajj3p0_TriplePFJet_v2',
    'HLT_VBF_DiPFJet125_45_Mjj720_Detajj3p0_v2',
    'HLT_VBF_DiPFJet45_Mjj500_Detajj2p5_Ele12_eta2p1_WPTight_Gsf_v1',
    'HLT_VBF_DiPFJet45_Mjj500_Detajj2p5_Ele17_eta2p1_WPTight_Gsf_v1',
    'HLT_VBF_DiPFJet45_Mjj500_Detajj2p5_MediumDeepTauPFTauHPS45_L2NN_eta2p1_v1',
    'HLT_VBF_DiPFJet45_Mjj500_Detajj2p5_Photon12_v1',
    'HLT_VBF_DiPFJet45_Mjj500_Detajj2p5_Photon17_v1',
    'HLT_VBF_DiPFJet50_Mjj500_Detajj2p5_Ele22_eta2p1_WPTight_Gsf_v1',
    'HLT_VBF_DiPFJet50_Mjj500_Detajj2p5_Photon22_v1',
    'HLT_VBF_DiPFJet70_40_Mjj600_Detajj2p5_DiPFJet60_JetMatchingFiveJet_v2',
    'HLT_VBF_DiPFJet70_40_Mjj600_Detajj2p5_DiPFJet60_JetMatchingQuadJet_v2',
    'HLT_VBF_DiPFJet70_40_Mjj600_Detajj2p5_DiPFJet60_JetMatchingSixJet_v2',
    'HLT_VBF_DiPFJet75_40_Mjj500_Detajj2p5_PFMET85_TriplePFJet_v2',
    'HLT_VBF_DiPFJet75_40_Mjj500_Detajj2p5_PFMET85_v2',
    'HLT_VBF_DiPFJet75_45_Mjj600_Detajj2p5_DiPFJet60_JetMatchingFiveJet_v1',
    'HLT_VBF_DiPFJet75_45_Mjj600_Detajj2p5_DiPFJet60_JetMatchingQuadJet_v1',
    'HLT_VBF_DiPFJet75_45_Mjj600_Detajj2p5_DiPFJet60_JetMatchingSixJet_v1',
    'HLT_VBF_DiPFJet80_45_Mjj500_Detajj2p5_PFMET85_TriplePFJet_v1',
    'HLT_VBF_DiPFJet80_45_Mjj500_Detajj2p5_PFMET85_v1',
    'HLT_VBF_DiPFJet90_40_Mjj600_Detajj2p5_Mu3_TrkIsoVVL_TriplePFJet_v2',
    'HLT_VBF_DiPFJet90_40_Mjj600_Detajj2p5_Mu3_TrkIsoVVL_v2',
    'HLT_VBF_DiPFJet95_45_Mjj600_Detajj2p5_Mu3_TrkIsoVVL_TriplePFJet_v1',
    'HLT_VBF_DiPFJet95_45_Mjj600_Detajj2p5_Mu3_TrkIsoVVL_v1',
    'HLT_VBF_DoubleMediumDeepTauPFTauHPS20_eta2p1_v5'
)


# stream ParkingVBF1

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingVBF1_datasetParkingVBF2_selector
streamParkingVBF1_datasetParkingVBF2_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingVBF1_datasetParkingVBF2_selector.l1tResults = cms.InputTag('')
streamParkingVBF1_datasetParkingVBF2_selector.throw      = cms.bool(False)
streamParkingVBF1_datasetParkingVBF2_selector.triggerConditions = cms.vstring(
    'HLT_DiJet110_35_Mjj650_PFMET110_v13',
    'HLT_DiJet110_35_Mjj650_PFMET120_v13',
    'HLT_DiJet110_35_Mjj650_PFMET130_v13',
    'HLT_DoublePFJets40_Mass500_MediumDeepTauPFTauHPS45_L2NN_MediumDeepTauPFTauHPS20_eta2p1_v4',
    'HLT_Mu10_TrkIsoVVL_DiPFJet40_DEta3p5_MJJ750_HTT350_PFMETNoMu60_v20',
    'HLT_Mu4_TrkIsoVVL_DiPFJet90_40_DEta3p5_MJJ750_HTT300_PFMETNoMu60_v20',
    'HLT_Mu8_TrkIsoVVL_DiPFJet40_DEta3p5_MJJ750_HTT300_PFMETNoMu60_v21',
    'HLT_QuadPFJet100_88_70_30_PNet1CvsAll0p5_VBF3Tight_v2',
    'HLT_QuadPFJet103_88_75_15_DoublePFBTagDeepJet_1p3_7p7_VBF1_v5',
    'HLT_QuadPFJet103_88_75_15_PFBTagDeepJet_1p3_VBF2_v5',
    'HLT_QuadPFJet105_88_75_30_PNet1CvsAll0p5_VBF3Tight_v2',
    'HLT_QuadPFJet105_88_76_15_DoublePFBTagDeepJet_1p3_7p7_VBF1_v5',
    'HLT_QuadPFJet105_88_76_15_PFBTagDeepJet_1p3_VBF2_v5',
    'HLT_QuadPFJet111_90_80_15_DoublePFBTagDeepJet_1p3_7p7_VBF1_v5',
    'HLT_QuadPFJet111_90_80_15_PFBTagDeepJet_1p3_VBF2_v5',
    'HLT_QuadPFJet111_90_80_30_PNet1CvsAll0p6_VBF3Tight_v2',
    'HLT_TripleJet110_35_35_Mjj650_PFMET110_v13',
    'HLT_TripleJet110_35_35_Mjj650_PFMET120_v13',
    'HLT_TripleJet110_35_35_Mjj650_PFMET130_v13',
    'HLT_VBF_DiPFJet105_40_Mjj1000_Detajj3p5_TriplePFJet_v2',
    'HLT_VBF_DiPFJet105_40_Mjj1000_Detajj3p5_v2',
    'HLT_VBF_DiPFJet110_40_Mjj1000_Detajj3p5_TriplePFJet_v1',
    'HLT_VBF_DiPFJet110_40_Mjj1000_Detajj3p5_v1',
    'HLT_VBF_DiPFJet125_45_Mjj1000_Detajj3p5_TriplePFJet_v1',
    'HLT_VBF_DiPFJet125_45_Mjj1000_Detajj3p5_v1',
    'HLT_VBF_DiPFJet125_45_Mjj720_Detajj3p0_TriplePFJet_v2',
    'HLT_VBF_DiPFJet125_45_Mjj720_Detajj3p0_v2',
    'HLT_VBF_DiPFJet45_Mjj500_Detajj2p5_Ele12_eta2p1_WPTight_Gsf_v1',
    'HLT_VBF_DiPFJet45_Mjj500_Detajj2p5_Ele17_eta2p1_WPTight_Gsf_v1',
    'HLT_VBF_DiPFJet45_Mjj500_Detajj2p5_MediumDeepTauPFTauHPS45_L2NN_eta2p1_v1',
    'HLT_VBF_DiPFJet45_Mjj500_Detajj2p5_Photon12_v1',
    'HLT_VBF_DiPFJet45_Mjj500_Detajj2p5_Photon17_v1',
    'HLT_VBF_DiPFJet50_Mjj500_Detajj2p5_Ele22_eta2p1_WPTight_Gsf_v1',
    'HLT_VBF_DiPFJet50_Mjj500_Detajj2p5_Photon22_v1',
    'HLT_VBF_DiPFJet70_40_Mjj600_Detajj2p5_DiPFJet60_JetMatchingFiveJet_v2',
    'HLT_VBF_DiPFJet70_40_Mjj600_Detajj2p5_DiPFJet60_JetMatchingQuadJet_v2',
    'HLT_VBF_DiPFJet70_40_Mjj600_Detajj2p5_DiPFJet60_JetMatchingSixJet_v2',
    'HLT_VBF_DiPFJet75_40_Mjj500_Detajj2p5_PFMET85_TriplePFJet_v2',
    'HLT_VBF_DiPFJet75_40_Mjj500_Detajj2p5_PFMET85_v2',
    'HLT_VBF_DiPFJet75_45_Mjj600_Detajj2p5_DiPFJet60_JetMatchingFiveJet_v1',
    'HLT_VBF_DiPFJet75_45_Mjj600_Detajj2p5_DiPFJet60_JetMatchingQuadJet_v1',
    'HLT_VBF_DiPFJet75_45_Mjj600_Detajj2p5_DiPFJet60_JetMatchingSixJet_v1',
    'HLT_VBF_DiPFJet80_45_Mjj500_Detajj2p5_PFMET85_TriplePFJet_v1',
    'HLT_VBF_DiPFJet80_45_Mjj500_Detajj2p5_PFMET85_v1',
    'HLT_VBF_DiPFJet90_40_Mjj600_Detajj2p5_Mu3_TrkIsoVVL_TriplePFJet_v2',
    'HLT_VBF_DiPFJet90_40_Mjj600_Detajj2p5_Mu3_TrkIsoVVL_v2',
    'HLT_VBF_DiPFJet95_45_Mjj600_Detajj2p5_Mu3_TrkIsoVVL_TriplePFJet_v1',
    'HLT_VBF_DiPFJet95_45_Mjj600_Detajj2p5_Mu3_TrkIsoVVL_v1',
    'HLT_VBF_DoubleMediumDeepTauPFTauHPS20_eta2p1_v5'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingVBF1_datasetParkingVBF3_selector
streamParkingVBF1_datasetParkingVBF3_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingVBF1_datasetParkingVBF3_selector.l1tResults = cms.InputTag('')
streamParkingVBF1_datasetParkingVBF3_selector.throw      = cms.bool(False)
streamParkingVBF1_datasetParkingVBF3_selector.triggerConditions = cms.vstring(
    'HLT_DiJet110_35_Mjj650_PFMET110_v13',
    'HLT_DiJet110_35_Mjj650_PFMET120_v13',
    'HLT_DiJet110_35_Mjj650_PFMET130_v13',
    'HLT_DoublePFJets40_Mass500_MediumDeepTauPFTauHPS45_L2NN_MediumDeepTauPFTauHPS20_eta2p1_v4',
    'HLT_Mu10_TrkIsoVVL_DiPFJet40_DEta3p5_MJJ750_HTT350_PFMETNoMu60_v20',
    'HLT_Mu4_TrkIsoVVL_DiPFJet90_40_DEta3p5_MJJ750_HTT300_PFMETNoMu60_v20',
    'HLT_Mu8_TrkIsoVVL_DiPFJet40_DEta3p5_MJJ750_HTT300_PFMETNoMu60_v21',
    'HLT_QuadPFJet100_88_70_30_PNet1CvsAll0p5_VBF3Tight_v2',
    'HLT_QuadPFJet103_88_75_15_DoublePFBTagDeepJet_1p3_7p7_VBF1_v5',
    'HLT_QuadPFJet103_88_75_15_PFBTagDeepJet_1p3_VBF2_v5',
    'HLT_QuadPFJet105_88_75_30_PNet1CvsAll0p5_VBF3Tight_v2',
    'HLT_QuadPFJet105_88_76_15_DoublePFBTagDeepJet_1p3_7p7_VBF1_v5',
    'HLT_QuadPFJet105_88_76_15_PFBTagDeepJet_1p3_VBF2_v5',
    'HLT_QuadPFJet111_90_80_15_DoublePFBTagDeepJet_1p3_7p7_VBF1_v5',
    'HLT_QuadPFJet111_90_80_15_PFBTagDeepJet_1p3_VBF2_v5',
    'HLT_QuadPFJet111_90_80_30_PNet1CvsAll0p6_VBF3Tight_v2',
    'HLT_TripleJet110_35_35_Mjj650_PFMET110_v13',
    'HLT_TripleJet110_35_35_Mjj650_PFMET120_v13',
    'HLT_TripleJet110_35_35_Mjj650_PFMET130_v13',
    'HLT_VBF_DiPFJet105_40_Mjj1000_Detajj3p5_TriplePFJet_v2',
    'HLT_VBF_DiPFJet105_40_Mjj1000_Detajj3p5_v2',
    'HLT_VBF_DiPFJet110_40_Mjj1000_Detajj3p5_TriplePFJet_v1',
    'HLT_VBF_DiPFJet110_40_Mjj1000_Detajj3p5_v1',
    'HLT_VBF_DiPFJet125_45_Mjj1000_Detajj3p5_TriplePFJet_v1',
    'HLT_VBF_DiPFJet125_45_Mjj1000_Detajj3p5_v1',
    'HLT_VBF_DiPFJet125_45_Mjj720_Detajj3p0_TriplePFJet_v2',
    'HLT_VBF_DiPFJet125_45_Mjj720_Detajj3p0_v2',
    'HLT_VBF_DiPFJet45_Mjj500_Detajj2p5_Ele12_eta2p1_WPTight_Gsf_v1',
    'HLT_VBF_DiPFJet45_Mjj500_Detajj2p5_Ele17_eta2p1_WPTight_Gsf_v1',
    'HLT_VBF_DiPFJet45_Mjj500_Detajj2p5_MediumDeepTauPFTauHPS45_L2NN_eta2p1_v1',
    'HLT_VBF_DiPFJet45_Mjj500_Detajj2p5_Photon12_v1',
    'HLT_VBF_DiPFJet45_Mjj500_Detajj2p5_Photon17_v1',
    'HLT_VBF_DiPFJet50_Mjj500_Detajj2p5_Ele22_eta2p1_WPTight_Gsf_v1',
    'HLT_VBF_DiPFJet50_Mjj500_Detajj2p5_Photon22_v1',
    'HLT_VBF_DiPFJet70_40_Mjj600_Detajj2p5_DiPFJet60_JetMatchingFiveJet_v2',
    'HLT_VBF_DiPFJet70_40_Mjj600_Detajj2p5_DiPFJet60_JetMatchingQuadJet_v2',
    'HLT_VBF_DiPFJet70_40_Mjj600_Detajj2p5_DiPFJet60_JetMatchingSixJet_v2',
    'HLT_VBF_DiPFJet75_40_Mjj500_Detajj2p5_PFMET85_TriplePFJet_v2',
    'HLT_VBF_DiPFJet75_40_Mjj500_Detajj2p5_PFMET85_v2',
    'HLT_VBF_DiPFJet75_45_Mjj600_Detajj2p5_DiPFJet60_JetMatchingFiveJet_v1',
    'HLT_VBF_DiPFJet75_45_Mjj600_Detajj2p5_DiPFJet60_JetMatchingQuadJet_v1',
    'HLT_VBF_DiPFJet75_45_Mjj600_Detajj2p5_DiPFJet60_JetMatchingSixJet_v1',
    'HLT_VBF_DiPFJet80_45_Mjj500_Detajj2p5_PFMET85_TriplePFJet_v1',
    'HLT_VBF_DiPFJet80_45_Mjj500_Detajj2p5_PFMET85_v1',
    'HLT_VBF_DiPFJet90_40_Mjj600_Detajj2p5_Mu3_TrkIsoVVL_TriplePFJet_v2',
    'HLT_VBF_DiPFJet90_40_Mjj600_Detajj2p5_Mu3_TrkIsoVVL_v2',
    'HLT_VBF_DiPFJet95_45_Mjj600_Detajj2p5_Mu3_TrkIsoVVL_TriplePFJet_v1',
    'HLT_VBF_DiPFJet95_45_Mjj600_Detajj2p5_Mu3_TrkIsoVVL_v1',
    'HLT_VBF_DoubleMediumDeepTauPFTauHPS20_eta2p1_v5'
)


# stream ParkingVBF2

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingVBF2_datasetParkingVBF4_selector
streamParkingVBF2_datasetParkingVBF4_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingVBF2_datasetParkingVBF4_selector.l1tResults = cms.InputTag('')
streamParkingVBF2_datasetParkingVBF4_selector.throw      = cms.bool(False)
streamParkingVBF2_datasetParkingVBF4_selector.triggerConditions = cms.vstring(
    'HLT_DiJet110_35_Mjj650_PFMET110_v13',
    'HLT_DiJet110_35_Mjj650_PFMET120_v13',
    'HLT_DiJet110_35_Mjj650_PFMET130_v13',
    'HLT_DoublePFJets40_Mass500_MediumDeepTauPFTauHPS45_L2NN_MediumDeepTauPFTauHPS20_eta2p1_v4',
    'HLT_Mu10_TrkIsoVVL_DiPFJet40_DEta3p5_MJJ750_HTT350_PFMETNoMu60_v20',
    'HLT_Mu4_TrkIsoVVL_DiPFJet90_40_DEta3p5_MJJ750_HTT300_PFMETNoMu60_v20',
    'HLT_Mu8_TrkIsoVVL_DiPFJet40_DEta3p5_MJJ750_HTT300_PFMETNoMu60_v21',
    'HLT_QuadPFJet100_88_70_30_PNet1CvsAll0p5_VBF3Tight_v2',
    'HLT_QuadPFJet103_88_75_15_DoublePFBTagDeepJet_1p3_7p7_VBF1_v5',
    'HLT_QuadPFJet103_88_75_15_PFBTagDeepJet_1p3_VBF2_v5',
    'HLT_QuadPFJet105_88_75_30_PNet1CvsAll0p5_VBF3Tight_v2',
    'HLT_QuadPFJet105_88_76_15_DoublePFBTagDeepJet_1p3_7p7_VBF1_v5',
    'HLT_QuadPFJet105_88_76_15_PFBTagDeepJet_1p3_VBF2_v5',
    'HLT_QuadPFJet111_90_80_15_DoublePFBTagDeepJet_1p3_7p7_VBF1_v5',
    'HLT_QuadPFJet111_90_80_15_PFBTagDeepJet_1p3_VBF2_v5',
    'HLT_QuadPFJet111_90_80_30_PNet1CvsAll0p6_VBF3Tight_v2',
    'HLT_TripleJet110_35_35_Mjj650_PFMET110_v13',
    'HLT_TripleJet110_35_35_Mjj650_PFMET120_v13',
    'HLT_TripleJet110_35_35_Mjj650_PFMET130_v13',
    'HLT_VBF_DiPFJet105_40_Mjj1000_Detajj3p5_TriplePFJet_v2',
    'HLT_VBF_DiPFJet105_40_Mjj1000_Detajj3p5_v2',
    'HLT_VBF_DiPFJet110_40_Mjj1000_Detajj3p5_TriplePFJet_v1',
    'HLT_VBF_DiPFJet110_40_Mjj1000_Detajj3p5_v1',
    'HLT_VBF_DiPFJet125_45_Mjj1000_Detajj3p5_TriplePFJet_v1',
    'HLT_VBF_DiPFJet125_45_Mjj1000_Detajj3p5_v1',
    'HLT_VBF_DiPFJet125_45_Mjj720_Detajj3p0_TriplePFJet_v2',
    'HLT_VBF_DiPFJet125_45_Mjj720_Detajj3p0_v2',
    'HLT_VBF_DiPFJet45_Mjj500_Detajj2p5_Ele12_eta2p1_WPTight_Gsf_v1',
    'HLT_VBF_DiPFJet45_Mjj500_Detajj2p5_Ele17_eta2p1_WPTight_Gsf_v1',
    'HLT_VBF_DiPFJet45_Mjj500_Detajj2p5_MediumDeepTauPFTauHPS45_L2NN_eta2p1_v1',
    'HLT_VBF_DiPFJet45_Mjj500_Detajj2p5_Photon12_v1',
    'HLT_VBF_DiPFJet45_Mjj500_Detajj2p5_Photon17_v1',
    'HLT_VBF_DiPFJet50_Mjj500_Detajj2p5_Ele22_eta2p1_WPTight_Gsf_v1',
    'HLT_VBF_DiPFJet50_Mjj500_Detajj2p5_Photon22_v1',
    'HLT_VBF_DiPFJet70_40_Mjj600_Detajj2p5_DiPFJet60_JetMatchingFiveJet_v2',
    'HLT_VBF_DiPFJet70_40_Mjj600_Detajj2p5_DiPFJet60_JetMatchingQuadJet_v2',
    'HLT_VBF_DiPFJet70_40_Mjj600_Detajj2p5_DiPFJet60_JetMatchingSixJet_v2',
    'HLT_VBF_DiPFJet75_40_Mjj500_Detajj2p5_PFMET85_TriplePFJet_v2',
    'HLT_VBF_DiPFJet75_40_Mjj500_Detajj2p5_PFMET85_v2',
    'HLT_VBF_DiPFJet75_45_Mjj600_Detajj2p5_DiPFJet60_JetMatchingFiveJet_v1',
    'HLT_VBF_DiPFJet75_45_Mjj600_Detajj2p5_DiPFJet60_JetMatchingQuadJet_v1',
    'HLT_VBF_DiPFJet75_45_Mjj600_Detajj2p5_DiPFJet60_JetMatchingSixJet_v1',
    'HLT_VBF_DiPFJet80_45_Mjj500_Detajj2p5_PFMET85_TriplePFJet_v1',
    'HLT_VBF_DiPFJet80_45_Mjj500_Detajj2p5_PFMET85_v1',
    'HLT_VBF_DiPFJet90_40_Mjj600_Detajj2p5_Mu3_TrkIsoVVL_TriplePFJet_v2',
    'HLT_VBF_DiPFJet90_40_Mjj600_Detajj2p5_Mu3_TrkIsoVVL_v2',
    'HLT_VBF_DiPFJet95_45_Mjj600_Detajj2p5_Mu3_TrkIsoVVL_TriplePFJet_v1',
    'HLT_VBF_DiPFJet95_45_Mjj600_Detajj2p5_Mu3_TrkIsoVVL_v1',
    'HLT_VBF_DoubleMediumDeepTauPFTauHPS20_eta2p1_v5'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingVBF2_datasetParkingVBF5_selector
streamParkingVBF2_datasetParkingVBF5_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingVBF2_datasetParkingVBF5_selector.l1tResults = cms.InputTag('')
streamParkingVBF2_datasetParkingVBF5_selector.throw      = cms.bool(False)
streamParkingVBF2_datasetParkingVBF5_selector.triggerConditions = cms.vstring(
    'HLT_DiJet110_35_Mjj650_PFMET110_v13',
    'HLT_DiJet110_35_Mjj650_PFMET120_v13',
    'HLT_DiJet110_35_Mjj650_PFMET130_v13',
    'HLT_DoublePFJets40_Mass500_MediumDeepTauPFTauHPS45_L2NN_MediumDeepTauPFTauHPS20_eta2p1_v4',
    'HLT_Mu10_TrkIsoVVL_DiPFJet40_DEta3p5_MJJ750_HTT350_PFMETNoMu60_v20',
    'HLT_Mu4_TrkIsoVVL_DiPFJet90_40_DEta3p5_MJJ750_HTT300_PFMETNoMu60_v20',
    'HLT_Mu8_TrkIsoVVL_DiPFJet40_DEta3p5_MJJ750_HTT300_PFMETNoMu60_v21',
    'HLT_QuadPFJet100_88_70_30_PNet1CvsAll0p5_VBF3Tight_v2',
    'HLT_QuadPFJet103_88_75_15_DoublePFBTagDeepJet_1p3_7p7_VBF1_v5',
    'HLT_QuadPFJet103_88_75_15_PFBTagDeepJet_1p3_VBF2_v5',
    'HLT_QuadPFJet105_88_75_30_PNet1CvsAll0p5_VBF3Tight_v2',
    'HLT_QuadPFJet105_88_76_15_DoublePFBTagDeepJet_1p3_7p7_VBF1_v5',
    'HLT_QuadPFJet105_88_76_15_PFBTagDeepJet_1p3_VBF2_v5',
    'HLT_QuadPFJet111_90_80_15_DoublePFBTagDeepJet_1p3_7p7_VBF1_v5',
    'HLT_QuadPFJet111_90_80_15_PFBTagDeepJet_1p3_VBF2_v5',
    'HLT_QuadPFJet111_90_80_30_PNet1CvsAll0p6_VBF3Tight_v2',
    'HLT_TripleJet110_35_35_Mjj650_PFMET110_v13',
    'HLT_TripleJet110_35_35_Mjj650_PFMET120_v13',
    'HLT_TripleJet110_35_35_Mjj650_PFMET130_v13',
    'HLT_VBF_DiPFJet105_40_Mjj1000_Detajj3p5_TriplePFJet_v2',
    'HLT_VBF_DiPFJet105_40_Mjj1000_Detajj3p5_v2',
    'HLT_VBF_DiPFJet110_40_Mjj1000_Detajj3p5_TriplePFJet_v1',
    'HLT_VBF_DiPFJet110_40_Mjj1000_Detajj3p5_v1',
    'HLT_VBF_DiPFJet125_45_Mjj1000_Detajj3p5_TriplePFJet_v1',
    'HLT_VBF_DiPFJet125_45_Mjj1000_Detajj3p5_v1',
    'HLT_VBF_DiPFJet125_45_Mjj720_Detajj3p0_TriplePFJet_v2',
    'HLT_VBF_DiPFJet125_45_Mjj720_Detajj3p0_v2',
    'HLT_VBF_DiPFJet45_Mjj500_Detajj2p5_Ele12_eta2p1_WPTight_Gsf_v1',
    'HLT_VBF_DiPFJet45_Mjj500_Detajj2p5_Ele17_eta2p1_WPTight_Gsf_v1',
    'HLT_VBF_DiPFJet45_Mjj500_Detajj2p5_MediumDeepTauPFTauHPS45_L2NN_eta2p1_v1',
    'HLT_VBF_DiPFJet45_Mjj500_Detajj2p5_Photon12_v1',
    'HLT_VBF_DiPFJet45_Mjj500_Detajj2p5_Photon17_v1',
    'HLT_VBF_DiPFJet50_Mjj500_Detajj2p5_Ele22_eta2p1_WPTight_Gsf_v1',
    'HLT_VBF_DiPFJet50_Mjj500_Detajj2p5_Photon22_v1',
    'HLT_VBF_DiPFJet70_40_Mjj600_Detajj2p5_DiPFJet60_JetMatchingFiveJet_v2',
    'HLT_VBF_DiPFJet70_40_Mjj600_Detajj2p5_DiPFJet60_JetMatchingQuadJet_v2',
    'HLT_VBF_DiPFJet70_40_Mjj600_Detajj2p5_DiPFJet60_JetMatchingSixJet_v2',
    'HLT_VBF_DiPFJet75_40_Mjj500_Detajj2p5_PFMET85_TriplePFJet_v2',
    'HLT_VBF_DiPFJet75_40_Mjj500_Detajj2p5_PFMET85_v2',
    'HLT_VBF_DiPFJet75_45_Mjj600_Detajj2p5_DiPFJet60_JetMatchingFiveJet_v1',
    'HLT_VBF_DiPFJet75_45_Mjj600_Detajj2p5_DiPFJet60_JetMatchingQuadJet_v1',
    'HLT_VBF_DiPFJet75_45_Mjj600_Detajj2p5_DiPFJet60_JetMatchingSixJet_v1',
    'HLT_VBF_DiPFJet80_45_Mjj500_Detajj2p5_PFMET85_TriplePFJet_v1',
    'HLT_VBF_DiPFJet80_45_Mjj500_Detajj2p5_PFMET85_v1',
    'HLT_VBF_DiPFJet90_40_Mjj600_Detajj2p5_Mu3_TrkIsoVVL_TriplePFJet_v2',
    'HLT_VBF_DiPFJet90_40_Mjj600_Detajj2p5_Mu3_TrkIsoVVL_v2',
    'HLT_VBF_DiPFJet95_45_Mjj600_Detajj2p5_Mu3_TrkIsoVVL_TriplePFJet_v1',
    'HLT_VBF_DiPFJet95_45_Mjj600_Detajj2p5_Mu3_TrkIsoVVL_v1',
    'HLT_VBF_DoubleMediumDeepTauPFTauHPS20_eta2p1_v5'
)


# stream ParkingVBF3

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingVBF3_datasetParkingVBF6_selector
streamParkingVBF3_datasetParkingVBF6_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingVBF3_datasetParkingVBF6_selector.l1tResults = cms.InputTag('')
streamParkingVBF3_datasetParkingVBF6_selector.throw      = cms.bool(False)
streamParkingVBF3_datasetParkingVBF6_selector.triggerConditions = cms.vstring(
    'HLT_DiJet110_35_Mjj650_PFMET110_v13',
    'HLT_DiJet110_35_Mjj650_PFMET120_v13',
    'HLT_DiJet110_35_Mjj650_PFMET130_v13',
    'HLT_DoublePFJets40_Mass500_MediumDeepTauPFTauHPS45_L2NN_MediumDeepTauPFTauHPS20_eta2p1_v4',
    'HLT_Mu10_TrkIsoVVL_DiPFJet40_DEta3p5_MJJ750_HTT350_PFMETNoMu60_v20',
    'HLT_Mu4_TrkIsoVVL_DiPFJet90_40_DEta3p5_MJJ750_HTT300_PFMETNoMu60_v20',
    'HLT_Mu8_TrkIsoVVL_DiPFJet40_DEta3p5_MJJ750_HTT300_PFMETNoMu60_v21',
    'HLT_QuadPFJet100_88_70_30_PNet1CvsAll0p5_VBF3Tight_v2',
    'HLT_QuadPFJet103_88_75_15_DoublePFBTagDeepJet_1p3_7p7_VBF1_v5',
    'HLT_QuadPFJet103_88_75_15_PFBTagDeepJet_1p3_VBF2_v5',
    'HLT_QuadPFJet105_88_75_30_PNet1CvsAll0p5_VBF3Tight_v2',
    'HLT_QuadPFJet105_88_76_15_DoublePFBTagDeepJet_1p3_7p7_VBF1_v5',
    'HLT_QuadPFJet105_88_76_15_PFBTagDeepJet_1p3_VBF2_v5',
    'HLT_QuadPFJet111_90_80_15_DoublePFBTagDeepJet_1p3_7p7_VBF1_v5',
    'HLT_QuadPFJet111_90_80_15_PFBTagDeepJet_1p3_VBF2_v5',
    'HLT_QuadPFJet111_90_80_30_PNet1CvsAll0p6_VBF3Tight_v2',
    'HLT_TripleJet110_35_35_Mjj650_PFMET110_v13',
    'HLT_TripleJet110_35_35_Mjj650_PFMET120_v13',
    'HLT_TripleJet110_35_35_Mjj650_PFMET130_v13',
    'HLT_VBF_DiPFJet105_40_Mjj1000_Detajj3p5_TriplePFJet_v2',
    'HLT_VBF_DiPFJet105_40_Mjj1000_Detajj3p5_v2',
    'HLT_VBF_DiPFJet110_40_Mjj1000_Detajj3p5_TriplePFJet_v1',
    'HLT_VBF_DiPFJet110_40_Mjj1000_Detajj3p5_v1',
    'HLT_VBF_DiPFJet125_45_Mjj1000_Detajj3p5_TriplePFJet_v1',
    'HLT_VBF_DiPFJet125_45_Mjj1000_Detajj3p5_v1',
    'HLT_VBF_DiPFJet125_45_Mjj720_Detajj3p0_TriplePFJet_v2',
    'HLT_VBF_DiPFJet125_45_Mjj720_Detajj3p0_v2',
    'HLT_VBF_DiPFJet45_Mjj500_Detajj2p5_Ele12_eta2p1_WPTight_Gsf_v1',
    'HLT_VBF_DiPFJet45_Mjj500_Detajj2p5_Ele17_eta2p1_WPTight_Gsf_v1',
    'HLT_VBF_DiPFJet45_Mjj500_Detajj2p5_MediumDeepTauPFTauHPS45_L2NN_eta2p1_v1',
    'HLT_VBF_DiPFJet45_Mjj500_Detajj2p5_Photon12_v1',
    'HLT_VBF_DiPFJet45_Mjj500_Detajj2p5_Photon17_v1',
    'HLT_VBF_DiPFJet50_Mjj500_Detajj2p5_Ele22_eta2p1_WPTight_Gsf_v1',
    'HLT_VBF_DiPFJet50_Mjj500_Detajj2p5_Photon22_v1',
    'HLT_VBF_DiPFJet70_40_Mjj600_Detajj2p5_DiPFJet60_JetMatchingFiveJet_v2',
    'HLT_VBF_DiPFJet70_40_Mjj600_Detajj2p5_DiPFJet60_JetMatchingQuadJet_v2',
    'HLT_VBF_DiPFJet70_40_Mjj600_Detajj2p5_DiPFJet60_JetMatchingSixJet_v2',
    'HLT_VBF_DiPFJet75_40_Mjj500_Detajj2p5_PFMET85_TriplePFJet_v2',
    'HLT_VBF_DiPFJet75_40_Mjj500_Detajj2p5_PFMET85_v2',
    'HLT_VBF_DiPFJet75_45_Mjj600_Detajj2p5_DiPFJet60_JetMatchingFiveJet_v1',
    'HLT_VBF_DiPFJet75_45_Mjj600_Detajj2p5_DiPFJet60_JetMatchingQuadJet_v1',
    'HLT_VBF_DiPFJet75_45_Mjj600_Detajj2p5_DiPFJet60_JetMatchingSixJet_v1',
    'HLT_VBF_DiPFJet80_45_Mjj500_Detajj2p5_PFMET85_TriplePFJet_v1',
    'HLT_VBF_DiPFJet80_45_Mjj500_Detajj2p5_PFMET85_v1',
    'HLT_VBF_DiPFJet90_40_Mjj600_Detajj2p5_Mu3_TrkIsoVVL_TriplePFJet_v2',
    'HLT_VBF_DiPFJet90_40_Mjj600_Detajj2p5_Mu3_TrkIsoVVL_v2',
    'HLT_VBF_DiPFJet95_45_Mjj600_Detajj2p5_Mu3_TrkIsoVVL_TriplePFJet_v1',
    'HLT_VBF_DiPFJet95_45_Mjj600_Detajj2p5_Mu3_TrkIsoVVL_v1',
    'HLT_VBF_DoubleMediumDeepTauPFTauHPS20_eta2p1_v5'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingVBF3_datasetParkingVBF7_selector
streamParkingVBF3_datasetParkingVBF7_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingVBF3_datasetParkingVBF7_selector.l1tResults = cms.InputTag('')
streamParkingVBF3_datasetParkingVBF7_selector.throw      = cms.bool(False)
streamParkingVBF3_datasetParkingVBF7_selector.triggerConditions = cms.vstring(
    'HLT_DiJet110_35_Mjj650_PFMET110_v13',
    'HLT_DiJet110_35_Mjj650_PFMET120_v13',
    'HLT_DiJet110_35_Mjj650_PFMET130_v13',
    'HLT_DoublePFJets40_Mass500_MediumDeepTauPFTauHPS45_L2NN_MediumDeepTauPFTauHPS20_eta2p1_v4',
    'HLT_Mu10_TrkIsoVVL_DiPFJet40_DEta3p5_MJJ750_HTT350_PFMETNoMu60_v20',
    'HLT_Mu4_TrkIsoVVL_DiPFJet90_40_DEta3p5_MJJ750_HTT300_PFMETNoMu60_v20',
    'HLT_Mu8_TrkIsoVVL_DiPFJet40_DEta3p5_MJJ750_HTT300_PFMETNoMu60_v21',
    'HLT_QuadPFJet100_88_70_30_PNet1CvsAll0p5_VBF3Tight_v2',
    'HLT_QuadPFJet103_88_75_15_DoublePFBTagDeepJet_1p3_7p7_VBF1_v5',
    'HLT_QuadPFJet103_88_75_15_PFBTagDeepJet_1p3_VBF2_v5',
    'HLT_QuadPFJet105_88_75_30_PNet1CvsAll0p5_VBF3Tight_v2',
    'HLT_QuadPFJet105_88_76_15_DoublePFBTagDeepJet_1p3_7p7_VBF1_v5',
    'HLT_QuadPFJet105_88_76_15_PFBTagDeepJet_1p3_VBF2_v5',
    'HLT_QuadPFJet111_90_80_15_DoublePFBTagDeepJet_1p3_7p7_VBF1_v5',
    'HLT_QuadPFJet111_90_80_15_PFBTagDeepJet_1p3_VBF2_v5',
    'HLT_QuadPFJet111_90_80_30_PNet1CvsAll0p6_VBF3Tight_v2',
    'HLT_TripleJet110_35_35_Mjj650_PFMET110_v13',
    'HLT_TripleJet110_35_35_Mjj650_PFMET120_v13',
    'HLT_TripleJet110_35_35_Mjj650_PFMET130_v13',
    'HLT_VBF_DiPFJet105_40_Mjj1000_Detajj3p5_TriplePFJet_v2',
    'HLT_VBF_DiPFJet105_40_Mjj1000_Detajj3p5_v2',
    'HLT_VBF_DiPFJet110_40_Mjj1000_Detajj3p5_TriplePFJet_v1',
    'HLT_VBF_DiPFJet110_40_Mjj1000_Detajj3p5_v1',
    'HLT_VBF_DiPFJet125_45_Mjj1000_Detajj3p5_TriplePFJet_v1',
    'HLT_VBF_DiPFJet125_45_Mjj1000_Detajj3p5_v1',
    'HLT_VBF_DiPFJet125_45_Mjj720_Detajj3p0_TriplePFJet_v2',
    'HLT_VBF_DiPFJet125_45_Mjj720_Detajj3p0_v2',
    'HLT_VBF_DiPFJet45_Mjj500_Detajj2p5_Ele12_eta2p1_WPTight_Gsf_v1',
    'HLT_VBF_DiPFJet45_Mjj500_Detajj2p5_Ele17_eta2p1_WPTight_Gsf_v1',
    'HLT_VBF_DiPFJet45_Mjj500_Detajj2p5_MediumDeepTauPFTauHPS45_L2NN_eta2p1_v1',
    'HLT_VBF_DiPFJet45_Mjj500_Detajj2p5_Photon12_v1',
    'HLT_VBF_DiPFJet45_Mjj500_Detajj2p5_Photon17_v1',
    'HLT_VBF_DiPFJet50_Mjj500_Detajj2p5_Ele22_eta2p1_WPTight_Gsf_v1',
    'HLT_VBF_DiPFJet50_Mjj500_Detajj2p5_Photon22_v1',
    'HLT_VBF_DiPFJet70_40_Mjj600_Detajj2p5_DiPFJet60_JetMatchingFiveJet_v2',
    'HLT_VBF_DiPFJet70_40_Mjj600_Detajj2p5_DiPFJet60_JetMatchingQuadJet_v2',
    'HLT_VBF_DiPFJet70_40_Mjj600_Detajj2p5_DiPFJet60_JetMatchingSixJet_v2',
    'HLT_VBF_DiPFJet75_40_Mjj500_Detajj2p5_PFMET85_TriplePFJet_v2',
    'HLT_VBF_DiPFJet75_40_Mjj500_Detajj2p5_PFMET85_v2',
    'HLT_VBF_DiPFJet75_45_Mjj600_Detajj2p5_DiPFJet60_JetMatchingFiveJet_v1',
    'HLT_VBF_DiPFJet75_45_Mjj600_Detajj2p5_DiPFJet60_JetMatchingQuadJet_v1',
    'HLT_VBF_DiPFJet75_45_Mjj600_Detajj2p5_DiPFJet60_JetMatchingSixJet_v1',
    'HLT_VBF_DiPFJet80_45_Mjj500_Detajj2p5_PFMET85_TriplePFJet_v1',
    'HLT_VBF_DiPFJet80_45_Mjj500_Detajj2p5_PFMET85_v1',
    'HLT_VBF_DiPFJet90_40_Mjj600_Detajj2p5_Mu3_TrkIsoVVL_TriplePFJet_v2',
    'HLT_VBF_DiPFJet90_40_Mjj600_Detajj2p5_Mu3_TrkIsoVVL_v2',
    'HLT_VBF_DiPFJet95_45_Mjj600_Detajj2p5_Mu3_TrkIsoVVL_TriplePFJet_v1',
    'HLT_VBF_DiPFJet95_45_Mjj600_Detajj2p5_Mu3_TrkIsoVVL_v1',
    'HLT_VBF_DoubleMediumDeepTauPFTauHPS20_eta2p1_v5'
)


# stream PhysicsCommissioning

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetCommissioning_selector
streamPhysicsCommissioning_datasetCommissioning_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetCommissioning_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetCommissioning_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetCommissioning_selector.triggerConditions = cms.vstring(
    'HLT_IsoTrackHB_v8',
    'HLT_IsoTrackHE_v8',
    'HLT_L1_CDC_SingleMu_3_er1p2_TOP120_DPHI2p618_3p142_v4',
    'HLT_PFJet40_GPUvsCPU_v2'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetCosmics_selector
streamPhysicsCommissioning_datasetCosmics_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetCosmics_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetCosmics_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetCosmics_selector.triggerConditions = cms.vstring('HLT_L1SingleMuCosmics_v3')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetHLTPhysics_selector
streamPhysicsCommissioning_datasetHLTPhysics_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetHLTPhysics_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetHLTPhysics_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetHLTPhysics_selector.triggerConditions = cms.vstring('HLT_Physics_v9')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetHcalNZS_selector
streamPhysicsCommissioning_datasetHcalNZS_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetHcalNZS_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetHcalNZS_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetHcalNZS_selector.triggerConditions = cms.vstring(
    'HLT_HcalNZS_v16',
    'HLT_HcalPhiSym_v18'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetMonteCarlo_selector
streamPhysicsCommissioning_datasetMonteCarlo_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetMonteCarlo_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetMonteCarlo_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetMonteCarlo_selector.triggerConditions = cms.vstring(
    'MC_AK4CaloJetsFromPV_v12',
    'MC_AK4CaloJets_v13',
    'MC_AK4PFJets_v21',
    'MC_AK8CaloHT_v12',
    'MC_AK8PFHT_v20',
    'MC_AK8PFJets_v21',
    'MC_AK8TrimPFJets_v21',
    'MC_CaloBTagDeepCSV_v12',
    'MC_CaloHT_v12',
    'MC_CaloMET_JetIdCleaned_v13',
    'MC_CaloMET_v12',
    'MC_CaloMHT_v12',
    'MC_Diphoton10_10_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass10_v17',
    'MC_DoubleEle5_CaloIdL_MW_v20',
    'MC_DoubleMuNoFiltersNoVtx_v11',
    'MC_DoubleMu_TrkIsoVVL_DZ_v15',
    'MC_Egamma_Open_Unseeded_v2',
    'MC_Egamma_Open_v2',
    'MC_Ele15_Ele10_CaloIdL_TrackIdL_IsoVL_DZ_v19',
    'MC_Ele5_WPTight_Gsf_v12',
    'MC_IsoMu_v19',
    'MC_PFBTagDeepCSV_v14',
    'MC_PFBTagDeepJet_v5',
    'MC_PFHT_v20',
    'MC_PFMET_v21',
    'MC_PFMHT_v20',
    'MC_QuadPFJet100_75_50_30_PNet2CvsL0p3And1CvsL0p5_VBF3Tight_v2',
    'MC_ReducedIterativeTracking_v16',
    'MC_Run3_PFScoutingPixelTracking_v20'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetNoBPTX_selector
streamPhysicsCommissioning_datasetNoBPTX_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetNoBPTX_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetNoBPTX_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetNoBPTX_selector.triggerConditions = cms.vstring(
    'HLT_CDC_L2cosmic_10_er1p0_v4',
    'HLT_CDC_L2cosmic_5p5_er1p0_v4',
    'HLT_L2Mu10_NoVertex_NoBPTX3BX_v8',
    'HLT_L2Mu10_NoVertex_NoBPTX_v9',
    'HLT_L2Mu40_NoVertex_3Sta_NoBPTX3BX_v8',
    'HLT_L2Mu45_NoVertex_3Sta_NoBPTX3BX_v7',
    'HLT_UncorrectedJetE30_NoBPTX3BX_v9',
    'HLT_UncorrectedJetE30_NoBPTX_v9',
    'HLT_UncorrectedJetE60_NoBPTX3BX_v9',
    'HLT_UncorrectedJetE70_NoBPTX3BX_v9'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetZeroBias_selector
streamPhysicsCommissioning_datasetZeroBias_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetZeroBias_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetZeroBias_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetZeroBias_selector.triggerConditions = cms.vstring(
    'HLT_Random_v3',
    'HLT_ZeroBias_Alignment_v3',
    'HLT_ZeroBias_FirstBXAfterTrain_v5',
    'HLT_ZeroBias_FirstCollisionAfterAbortGap_v7',
    'HLT_ZeroBias_FirstCollisionInTrain_v6',
    'HLT_ZeroBias_IsolatedBunches_v7',
    'HLT_ZeroBias_LastCollisionInTrain_v5',
    'HLT_ZeroBias_v8'
)


# stream PhysicsDispJetBTagMuEGTau

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsDispJetBTagMuEGTau_datasetBTagMu_selector
streamPhysicsDispJetBTagMuEGTau_datasetBTagMu_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsDispJetBTagMuEGTau_datasetBTagMu_selector.l1tResults = cms.InputTag('')
streamPhysicsDispJetBTagMuEGTau_datasetBTagMu_selector.throw      = cms.bool(False)
streamPhysicsDispJetBTagMuEGTau_datasetBTagMu_selector.triggerConditions = cms.vstring(
    'HLT_BTagMu_AK4DiJet110_Mu5_v17',
    'HLT_BTagMu_AK4DiJet170_Mu5_v16',
    'HLT_BTagMu_AK4DiJet20_Mu5_v17',
    'HLT_BTagMu_AK4DiJet40_Mu5_v17',
    'HLT_BTagMu_AK4DiJet70_Mu5_v17',
    'HLT_BTagMu_AK4Jet300_Mu5_v16',
    'HLT_BTagMu_AK8DiJet170_Mu5_v13',
    'HLT_BTagMu_AK8Jet170_DoubleMu5_v6',
    'HLT_BTagMu_AK8Jet300_Mu5_v16'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsDispJetBTagMuEGTau_datasetDisplacedJet_selector
streamPhysicsDispJetBTagMuEGTau_datasetDisplacedJet_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsDispJetBTagMuEGTau_datasetDisplacedJet_selector.l1tResults = cms.InputTag('')
streamPhysicsDispJetBTagMuEGTau_datasetDisplacedJet_selector.throw      = cms.bool(False)
streamPhysicsDispJetBTagMuEGTau_datasetDisplacedJet_selector.triggerConditions = cms.vstring(
    'HLT_CaloMET60_DTCluster50_v5',
    'HLT_CaloMET60_DTClusterNoMB1S50_v5',
    'HLT_HT170_L1SingleLLPJet_DisplacedDijet40_DisplacedTrack_v5',
    'HLT_HT200_L1SingleLLPJet_DelayedJet40_DoubleDelay0p5nsTrackless_v5',
    'HLT_HT200_L1SingleLLPJet_DelayedJet40_DoubleDelay1nsInclusive_v5',
    'HLT_HT200_L1SingleLLPJet_DelayedJet40_SingleDelay1nsTrackless_v5',
    'HLT_HT200_L1SingleLLPJet_DelayedJet40_SingleDelay2nsInclusive_v5',
    'HLT_HT200_L1SingleLLPJet_DisplacedDijet35_Inclusive1PtrkShortSig5_v5',
    'HLT_HT200_L1SingleLLPJet_DisplacedDijet40_DisplacedTrack_v5',
    'HLT_HT200_L1SingleLLPJet_DisplacedDijet40_Inclusive1PtrkShortSig5_v5',
    'HLT_HT200_L1SingleLLPJet_DisplacedDijet60_DisplacedTrack_v5',
    'HLT_HT240_L1SingleLLPJet_DisplacedDijet40_Inclusive1PtrkShortSig5_v2',
    'HLT_HT270_L1SingleLLPJet_DisplacedDijet40_DisplacedTrack_v5',
    'HLT_HT280_L1SingleLLPJet_DisplacedDijet40_Inclusive1PtrkShortSig5_v2',
    'HLT_HT320_L1SingleLLPJet_DisplacedDijet60_Inclusive_v5',
    'HLT_HT350_DelayedJet40_SingleDelay3nsInclusive_v1',
    'HLT_HT350_DelayedJet40_SingleDelay3p25nsInclusive_v1',
    'HLT_HT350_DelayedJet40_SingleDelay3p5nsInclusive_v1',
    'HLT_HT350_v1',
    'HLT_HT400_DisplacedDijet40_DisplacedTrack_v17',
    'HLT_HT420_L1SingleLLPJet_DisplacedDijet60_Inclusive_v5',
    'HLT_HT425_v13',
    'HLT_HT430_DelayedJet40_DoubleDelay0p5nsInclusive_v4',
    'HLT_HT430_DelayedJet40_DoubleDelay0p5nsTrackless_v5',
    'HLT_HT430_DelayedJet40_DoubleDelay0p75nsTrackless_v1',
    'HLT_HT430_DelayedJet40_DoubleDelay1nsInclusive_v5',
    'HLT_HT430_DelayedJet40_DoubleDelay1nsTrackless_v1',
    'HLT_HT430_DelayedJet40_DoubleDelay1p25nsInclusive_v1',
    'HLT_HT430_DelayedJet40_DoubleDelay1p5nsInclusive_v1',
    'HLT_HT430_DelayedJet40_SingleDelay0p5nsInclusive_v3',
    'HLT_HT430_DelayedJet40_SingleDelay0p5nsTrackless_v3',
    'HLT_HT430_DelayedJet40_SingleDelay1nsInclusive_v3',
    'HLT_HT430_DelayedJet40_SingleDelay1nsTrackless_v5',
    'HLT_HT430_DelayedJet40_SingleDelay1p25nsTrackless_v1',
    'HLT_HT430_DelayedJet40_SingleDelay1p5nsInclusive_v3',
    'HLT_HT430_DelayedJet40_SingleDelay1p5nsTrackless_v1',
    'HLT_HT430_DelayedJet40_SingleDelay2nsInclusive_v5',
    'HLT_HT430_DelayedJet40_SingleDelay2p25nsInclusive_v1',
    'HLT_HT430_DelayedJet40_SingleDelay2p5nsInclusive_v1',
    'HLT_HT550_DisplacedDijet60_Inclusive_v17',
    'HLT_L1MET_DTCluster50_v5',
    'HLT_L1MET_DTClusterNoMB1S50_v5',
    'HLT_L1Mu6HT240_v3',
    'HLT_L1SingleLLPJet_v2',
    'HLT_L1Tau_DelayedJet40_DoubleDelay0p5nsTrackless_v3',
    'HLT_L1Tau_DelayedJet40_DoubleDelay0p75nsInclusive_v3',
    'HLT_L1Tau_DelayedJet40_DoubleDelay1nsTrackless_v3',
    'HLT_L1Tau_DelayedJet40_DoubleDelay1p25nsInclusive_v3',
    'HLT_L1Tau_DelayedJet40_DoubleDelay1p25nsTrackless_v1',
    'HLT_L1Tau_DelayedJet40_DoubleDelay1p5nsInclusive_v1',
    'HLT_L1Tau_DelayedJet40_DoubleDelay1p5nsTrackless_v1',
    'HLT_L1Tau_DelayedJet40_DoubleDelay1p75nsInclusive_v1',
    'HLT_L1Tau_DelayedJet40_SingleDelay2p5nsTrackless_v3',
    'HLT_L1Tau_DelayedJet40_SingleDelay2p75nsTrackless_v1',
    'HLT_L1Tau_DelayedJet40_SingleDelay3nsTrackless_v1',
    'HLT_L1Tau_DelayedJet40_SingleDelay3p5nsInclusive_v3',
    'HLT_L1Tau_DelayedJet40_SingleDelay3p75nsInclusive_v1',
    'HLT_L1Tau_DelayedJet40_SingleDelay4nsInclusive_v1',
    'HLT_Mu6HT240_DisplacedDijet30_Inclusive1PtrkShortSig5_DisplacedLoose_v5',
    'HLT_Mu6HT240_DisplacedDijet35_Inclusive0PtrkShortSig5_v5',
    'HLT_Mu6HT240_DisplacedDijet35_Inclusive1PtrkShortSig5_DisplacedLoose_v5',
    'HLT_Mu6HT240_DisplacedDijet40_Inclusive0PtrkShortSig5_v5',
    'HLT_Mu6HT240_DisplacedDijet40_Inclusive1PtrkShortSig5_DisplacedLoose_v5',
    'HLT_Mu6HT240_DisplacedDijet45_Inclusive0PtrkShortSig5_v5',
    'HLT_Mu6HT240_DisplacedDijet50_Inclusive0PtrkShortSig5_v5',
    'HLT_PFJet200_TimeGt2p5ns_v2',
    'HLT_PFJet200_TimeLtNeg2p5ns_v2'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsDispJetBTagMuEGTau_datasetMuonEG_selector
streamPhysicsDispJetBTagMuEGTau_datasetMuonEG_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsDispJetBTagMuEGTau_datasetMuonEG_selector.l1tResults = cms.InputTag('')
streamPhysicsDispJetBTagMuEGTau_datasetMuonEG_selector.throw      = cms.bool(False)
streamPhysicsDispJetBTagMuEGTau_datasetMuonEG_selector.triggerConditions = cms.vstring(
    'HLT_DiMu4_Ele9_CaloIdL_TrackIdL_DZ_Mass3p8_v21',
    'HLT_DiMu9_Ele9_CaloIdL_TrackIdL_DZ_v21',
    'HLT_DiMu9_Ele9_CaloIdL_TrackIdL_v21',
    'HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_v19',
    'HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_v11',
    'HLT_Mu17_Photon30_IsoCaloId_v10',
    'HLT_Mu20NoFiltersNoVtxDisplaced_Photon20_CaloCustomId_v5',
    'HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v19',
    'HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_v11',
    'HLT_Mu27_Ele37_CaloIdL_MW_v9',
    'HLT_Mu37_Ele27_CaloIdL_MW_v9',
    'HLT_Mu38NoFiltersNoVtxDisplaced_Photon38_CaloIdL_v5',
    'HLT_Mu43NoFiltersNoVtxDisplaced_Photon43_CaloIdL_v5',
    'HLT_Mu43NoFiltersNoVtx_Photon43_CaloIdL_v9',
    'HLT_Mu48NoFiltersNoVtx_Photon48_CaloIdL_v9',
    'HLT_Mu8_DiEle12_CaloIdL_TrackIdL_DZ_v22',
    'HLT_Mu8_DiEle12_CaloIdL_TrackIdL_v22',
    'HLT_Mu8_Ele8_CaloIdM_TrackIdM_Mass8_PFHT350_DZ_v23',
    'HLT_Mu8_Ele8_CaloIdM_TrackIdM_Mass8_PFHT350_v23',
    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_CaloDiJet30_v5',
    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_DoubleAK4PFJet60_30_PNet2BTagMean0p50_v2',
    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_DoubleAK4PFJet60_30_v2',
    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFDiJet30_PFBtagDeepJet_1p5_v5',
    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFDiJet30_PNet2BTagMean0p50_v1',
    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFDiJet30_v5',
    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFHT280_QuadPFJet30_PNet2BTagMean0p55_v1',
    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFHT280_QuadPFJet30_v1',
    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFHT280_v1',
    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_v17',
    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_v15'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsDispJetBTagMuEGTau_datasetTau_selector
streamPhysicsDispJetBTagMuEGTau_datasetTau_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsDispJetBTagMuEGTau_datasetTau_selector.l1tResults = cms.InputTag('')
streamPhysicsDispJetBTagMuEGTau_datasetTau_selector.throw      = cms.bool(False)
streamPhysicsDispJetBTagMuEGTau_datasetTau_selector.triggerConditions = cms.vstring(
    'HLT_DoubleMediumChargedIsoDisplacedPFTauHPS32_Trk1_eta2p1_v5',
    'HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_OneProng_M5to80_v2',
    'HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60_v4',
    'HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet75_v4',
    'HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1_v4',
    'HLT_LooseDeepTauPFTauHPS180_L2NN_eta2p1_v5'
)


# stream PhysicsEGamma0

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsEGamma0_datasetEGamma0_selector
streamPhysicsEGamma0_datasetEGamma0_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsEGamma0_datasetEGamma0_selector.l1tResults = cms.InputTag('')
streamPhysicsEGamma0_datasetEGamma0_selector.throw      = cms.bool(False)
streamPhysicsEGamma0_datasetEGamma0_selector.triggerConditions = cms.vstring(
    'HLT_DiEle27_WPTightCaloOnly_L1DoubleEG_v8',
    'HLT_DiPhoton10Time1ns_v4',
    'HLT_DiPhoton10Time1p2ns_v4',
    'HLT_DiPhoton10Time1p4ns_v4',
    'HLT_DiPhoton10Time1p6ns_v4',
    'HLT_DiPhoton10Time1p8ns_v4',
    'HLT_DiPhoton10Time2ns_v4',
    'HLT_DiPhoton10_CaloIdL_v4',
    'HLT_DiPhoton10sminlt0p12_v4',
    'HLT_DiPhoton10sminlt0p1_v4',
    'HLT_DiSC30_18_EIso_AND_HE_Mass70_v18',
    'HLT_Diphoton20_14_eta1p5_R9IdL_AND_HET_AND_IsoTCaloIdT_v4',
    'HLT_Diphoton20_14_eta1p5_R9IdL_AND_HE_AND_IsoTCaloIdT_v4',
    'HLT_Diphoton22_14_eta1p5_R9IdL_AND_HET_AND_IsoTCaloIdT_v4',
    'HLT_Diphoton22_14_eta1p5_R9IdL_AND_HE_AND_IsoTCaloIdT_v4',
    'HLT_Diphoton24_14_eta1p5_R9IdL_AND_HET_AND_IsoTCaloIdT_v4',
    'HLT_Diphoton24_16_eta1p5_R9IdL_AND_HET_AND_IsoTCaloIdT_v4',
    'HLT_Diphoton30_18_R9IdL_AND_HE_AND_IsoCaloId_Mass55_v5',
    'HLT_Diphoton30_18_R9IdL_AND_HE_AND_IsoCaloId_v5',
    'HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass90_v17',
    'HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass95_v17',
    'HLT_DoubleEle25_CaloIdL_MW_v9',
    'HLT_DoubleEle27_CaloIdL_MW_v9',
    'HLT_DoubleEle33_CaloIdL_MW_v22',
    'HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_DZ_PFHT350_v24',
    'HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_PFHT350_v24',
    'HLT_DoublePhoton33_CaloIdL_v11',
    'HLT_DoublePhoton70_v11',
    'HLT_DoublePhoton85_v19',
    'HLT_ECALHT800_v14',
    'HLT_Ele115_CaloIdVT_GsfTrkIdT_v19',
    'HLT_Ele12_CaloIdL_TrackIdL_IsoVL_PFJet30_v22',
    'HLT_Ele135_CaloIdVT_GsfTrkIdT_v12',
    'HLT_Ele15_IsoVVVL_PFHT450_PFMET50_v20',
    'HLT_Ele15_IsoVVVL_PFHT450_v20',
    'HLT_Ele15_IsoVVVL_PFHT600_v24',
    'HLT_Ele16_Ele12_Ele8_CaloIdL_TrackIdL_v13',
    'HLT_Ele17_CaloIdM_TrackIdM_PFJet30_v20',
    'HLT_Ele23_CaloIdL_TrackIdL_IsoVL_PFJet30_v22',
    'HLT_Ele23_CaloIdM_TrackIdM_PFJet30_v22',
    'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v23',
    'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_v23',
    'HLT_Ele24_eta2p1_WPTight_Gsf_LooseDeepTauPFTauHPS30_eta2p1_CrossL1_v5',
    'HLT_Ele28_HighEta_SC20_Mass55_v17',
    'HLT_Ele28_eta2p1_WPTight_Gsf_HT150_v17',
    'HLT_Ele30_WPTight_Gsf_v5',
    'HLT_Ele30_eta2p1_WPTight_Gsf_CentralPFJet35_EleCleaned_v17',
    'HLT_Ele32_WPTight_Gsf_L1DoubleEG_v13',
    'HLT_Ele32_WPTight_Gsf_v19',
    'HLT_Ele35_WPTight_Gsf_v13',
    'HLT_Ele38_WPTight_Gsf_v13',
    'HLT_Ele40_WPTight_Gsf_v13',
    'HLT_Ele50_CaloIdVT_GsfTrkIdT_AK8PFJet220_SoftDropMass40_PNetBB0p06_v1',
    'HLT_Ele50_CaloIdVT_GsfTrkIdT_AK8PFJet220_SoftDropMass40_v4',
    'HLT_Ele50_CaloIdVT_GsfTrkIdT_AK8PFJet230_SoftDropMass40_PNetBB0p06_v1',
    'HLT_Ele50_CaloIdVT_GsfTrkIdT_AK8PFJet230_SoftDropMass40_PNetBB0p10_v1',
    'HLT_Ele50_CaloIdVT_GsfTrkIdT_AK8PFJet230_SoftDropMass40_v4',
    'HLT_Ele50_CaloIdVT_GsfTrkIdT_PFJet165_v22',
    'HLT_Ele50_IsoVVVL_PFHT450_v20',
    'HLT_Ele8_CaloIdL_TrackIdL_IsoVL_PFJet30_v20',
    'HLT_Ele8_CaloIdM_TrackIdM_PFJet30_v22',
    'HLT_Photon100EBHE10_v6',
    'HLT_Photon110EB_TightID_TightIso_v6',
    'HLT_Photon120_R9Id90_HE10_IsoM_v18',
    'HLT_Photon120_v17',
    'HLT_Photon130EB_TightID_TightIso_v2',
    'HLT_Photon150EB_TightID_TightIso_v2',
    'HLT_Photon150_v11',
    'HLT_Photon165_R9Id90_HE10_IsoM_v19',
    'HLT_Photon175EB_TightID_TightIso_v2',
    'HLT_Photon175_v19',
    'HLT_Photon200EB_TightID_TightIso_v2',
    'HLT_Photon200_v18',
    'HLT_Photon20_HoverELoose_v14',
    'HLT_Photon300_NoHE_v17',
    'HLT_Photon30EB_TightID_TightIso_v5',
    'HLT_Photon30_HoverELoose_v14',
    'HLT_Photon32_OneProng32_M50To105_v2',
    'HLT_Photon33_v9',
    'HLT_Photon35_TwoProngs35_v5',
    'HLT_Photon50EB_TightID_TightIso_v2',
    'HLT_Photon50_R9Id90_HE10_IsoM_v18',
    'HLT_Photon50_TimeGt2p5ns_v1',
    'HLT_Photon50_TimeLtNeg2p5ns_v1',
    'HLT_Photon50_v17',
    'HLT_Photon60_R9Id90_CaloIdL_IsoL_DisplacedIdL_PFHT350_v2',
    'HLT_Photon60_R9Id90_CaloIdL_IsoL_DisplacedIdL_PFHT380_v2',
    'HLT_Photon60_R9Id90_CaloIdL_IsoL_DisplacedIdL_PFHT400_v2',
    'HLT_Photon60_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ300DEta3_v1',
    'HLT_Photon75EB_TightID_TightIso_v2',
    'HLT_Photon75_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ300DEta3_v9',
    'HLT_Photon75_R9Id90_HE10_IsoM_v18',
    'HLT_Photon75_v17',
    'HLT_Photon90EB_TightID_TightIso_v2',
    'HLT_Photon90_R9Id90_HE10_IsoM_v18',
    'HLT_Photon90_v17'
)


# stream PhysicsEGamma1

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsEGamma1_datasetEGamma1_selector
streamPhysicsEGamma1_datasetEGamma1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsEGamma1_datasetEGamma1_selector.l1tResults = cms.InputTag('')
streamPhysicsEGamma1_datasetEGamma1_selector.throw      = cms.bool(False)
streamPhysicsEGamma1_datasetEGamma1_selector.triggerConditions = cms.vstring(
    'HLT_DiEle27_WPTightCaloOnly_L1DoubleEG_v8',
    'HLT_DiPhoton10Time1ns_v4',
    'HLT_DiPhoton10Time1p2ns_v4',
    'HLT_DiPhoton10Time1p4ns_v4',
    'HLT_DiPhoton10Time1p6ns_v4',
    'HLT_DiPhoton10Time1p8ns_v4',
    'HLT_DiPhoton10Time2ns_v4',
    'HLT_DiPhoton10_CaloIdL_v4',
    'HLT_DiPhoton10sminlt0p12_v4',
    'HLT_DiPhoton10sminlt0p1_v4',
    'HLT_DiSC30_18_EIso_AND_HE_Mass70_v18',
    'HLT_Diphoton20_14_eta1p5_R9IdL_AND_HET_AND_IsoTCaloIdT_v4',
    'HLT_Diphoton20_14_eta1p5_R9IdL_AND_HE_AND_IsoTCaloIdT_v4',
    'HLT_Diphoton22_14_eta1p5_R9IdL_AND_HET_AND_IsoTCaloIdT_v4',
    'HLT_Diphoton22_14_eta1p5_R9IdL_AND_HE_AND_IsoTCaloIdT_v4',
    'HLT_Diphoton24_14_eta1p5_R9IdL_AND_HET_AND_IsoTCaloIdT_v4',
    'HLT_Diphoton24_16_eta1p5_R9IdL_AND_HET_AND_IsoTCaloIdT_v4',
    'HLT_Diphoton30_18_R9IdL_AND_HE_AND_IsoCaloId_Mass55_v5',
    'HLT_Diphoton30_18_R9IdL_AND_HE_AND_IsoCaloId_v5',
    'HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass90_v17',
    'HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass95_v17',
    'HLT_DoubleEle25_CaloIdL_MW_v9',
    'HLT_DoubleEle27_CaloIdL_MW_v9',
    'HLT_DoubleEle33_CaloIdL_MW_v22',
    'HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_DZ_PFHT350_v24',
    'HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_PFHT350_v24',
    'HLT_DoublePhoton33_CaloIdL_v11',
    'HLT_DoublePhoton70_v11',
    'HLT_DoublePhoton85_v19',
    'HLT_ECALHT800_v14',
    'HLT_Ele115_CaloIdVT_GsfTrkIdT_v19',
    'HLT_Ele12_CaloIdL_TrackIdL_IsoVL_PFJet30_v22',
    'HLT_Ele135_CaloIdVT_GsfTrkIdT_v12',
    'HLT_Ele15_IsoVVVL_PFHT450_PFMET50_v20',
    'HLT_Ele15_IsoVVVL_PFHT450_v20',
    'HLT_Ele15_IsoVVVL_PFHT600_v24',
    'HLT_Ele16_Ele12_Ele8_CaloIdL_TrackIdL_v13',
    'HLT_Ele17_CaloIdM_TrackIdM_PFJet30_v20',
    'HLT_Ele23_CaloIdL_TrackIdL_IsoVL_PFJet30_v22',
    'HLT_Ele23_CaloIdM_TrackIdM_PFJet30_v22',
    'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v23',
    'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_v23',
    'HLT_Ele24_eta2p1_WPTight_Gsf_LooseDeepTauPFTauHPS30_eta2p1_CrossL1_v5',
    'HLT_Ele28_HighEta_SC20_Mass55_v17',
    'HLT_Ele28_eta2p1_WPTight_Gsf_HT150_v17',
    'HLT_Ele30_WPTight_Gsf_v5',
    'HLT_Ele30_eta2p1_WPTight_Gsf_CentralPFJet35_EleCleaned_v17',
    'HLT_Ele32_WPTight_Gsf_L1DoubleEG_v13',
    'HLT_Ele32_WPTight_Gsf_v19',
    'HLT_Ele35_WPTight_Gsf_v13',
    'HLT_Ele38_WPTight_Gsf_v13',
    'HLT_Ele40_WPTight_Gsf_v13',
    'HLT_Ele50_CaloIdVT_GsfTrkIdT_AK8PFJet220_SoftDropMass40_PNetBB0p06_v1',
    'HLT_Ele50_CaloIdVT_GsfTrkIdT_AK8PFJet220_SoftDropMass40_v4',
    'HLT_Ele50_CaloIdVT_GsfTrkIdT_AK8PFJet230_SoftDropMass40_PNetBB0p06_v1',
    'HLT_Ele50_CaloIdVT_GsfTrkIdT_AK8PFJet230_SoftDropMass40_PNetBB0p10_v1',
    'HLT_Ele50_CaloIdVT_GsfTrkIdT_AK8PFJet230_SoftDropMass40_v4',
    'HLT_Ele50_CaloIdVT_GsfTrkIdT_PFJet165_v22',
    'HLT_Ele50_IsoVVVL_PFHT450_v20',
    'HLT_Ele8_CaloIdL_TrackIdL_IsoVL_PFJet30_v20',
    'HLT_Ele8_CaloIdM_TrackIdM_PFJet30_v22',
    'HLT_Photon100EBHE10_v6',
    'HLT_Photon110EB_TightID_TightIso_v6',
    'HLT_Photon120_R9Id90_HE10_IsoM_v18',
    'HLT_Photon120_v17',
    'HLT_Photon130EB_TightID_TightIso_v2',
    'HLT_Photon150EB_TightID_TightIso_v2',
    'HLT_Photon150_v11',
    'HLT_Photon165_R9Id90_HE10_IsoM_v19',
    'HLT_Photon175EB_TightID_TightIso_v2',
    'HLT_Photon175_v19',
    'HLT_Photon200EB_TightID_TightIso_v2',
    'HLT_Photon200_v18',
    'HLT_Photon20_HoverELoose_v14',
    'HLT_Photon300_NoHE_v17',
    'HLT_Photon30EB_TightID_TightIso_v5',
    'HLT_Photon30_HoverELoose_v14',
    'HLT_Photon32_OneProng32_M50To105_v2',
    'HLT_Photon33_v9',
    'HLT_Photon35_TwoProngs35_v5',
    'HLT_Photon50EB_TightID_TightIso_v2',
    'HLT_Photon50_R9Id90_HE10_IsoM_v18',
    'HLT_Photon50_TimeGt2p5ns_v1',
    'HLT_Photon50_TimeLtNeg2p5ns_v1',
    'HLT_Photon50_v17',
    'HLT_Photon60_R9Id90_CaloIdL_IsoL_DisplacedIdL_PFHT350_v2',
    'HLT_Photon60_R9Id90_CaloIdL_IsoL_DisplacedIdL_PFHT380_v2',
    'HLT_Photon60_R9Id90_CaloIdL_IsoL_DisplacedIdL_PFHT400_v2',
    'HLT_Photon60_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ300DEta3_v1',
    'HLT_Photon75EB_TightID_TightIso_v2',
    'HLT_Photon75_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ300DEta3_v9',
    'HLT_Photon75_R9Id90_HE10_IsoM_v18',
    'HLT_Photon75_v17',
    'HLT_Photon90EB_TightID_TightIso_v2',
    'HLT_Photon90_R9Id90_HE10_IsoM_v18',
    'HLT_Photon90_v17'
)


# stream PhysicsHLTPhysics0

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHLTPhysics0_datasetEphemeralHLTPhysics0_selector
streamPhysicsHLTPhysics0_datasetEphemeralHLTPhysics0_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHLTPhysics0_datasetEphemeralHLTPhysics0_selector.l1tResults = cms.InputTag('')
streamPhysicsHLTPhysics0_datasetEphemeralHLTPhysics0_selector.throw      = cms.bool(False)
streamPhysicsHLTPhysics0_datasetEphemeralHLTPhysics0_selector.triggerConditions = cms.vstring('HLT_EphemeralPhysics_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHLTPhysics0_datasetEphemeralHLTPhysics1_selector
streamPhysicsHLTPhysics0_datasetEphemeralHLTPhysics1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHLTPhysics0_datasetEphemeralHLTPhysics1_selector.l1tResults = cms.InputTag('')
streamPhysicsHLTPhysics0_datasetEphemeralHLTPhysics1_selector.throw      = cms.bool(False)
streamPhysicsHLTPhysics0_datasetEphemeralHLTPhysics1_selector.triggerConditions = cms.vstring('HLT_EphemeralPhysics_v4')


# stream PhysicsHLTPhysics1

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHLTPhysics1_datasetEphemeralHLTPhysics2_selector
streamPhysicsHLTPhysics1_datasetEphemeralHLTPhysics2_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHLTPhysics1_datasetEphemeralHLTPhysics2_selector.l1tResults = cms.InputTag('')
streamPhysicsHLTPhysics1_datasetEphemeralHLTPhysics2_selector.throw      = cms.bool(False)
streamPhysicsHLTPhysics1_datasetEphemeralHLTPhysics2_selector.triggerConditions = cms.vstring('HLT_EphemeralPhysics_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHLTPhysics1_datasetEphemeralHLTPhysics3_selector
streamPhysicsHLTPhysics1_datasetEphemeralHLTPhysics3_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHLTPhysics1_datasetEphemeralHLTPhysics3_selector.l1tResults = cms.InputTag('')
streamPhysicsHLTPhysics1_datasetEphemeralHLTPhysics3_selector.throw      = cms.bool(False)
streamPhysicsHLTPhysics1_datasetEphemeralHLTPhysics3_selector.triggerConditions = cms.vstring('HLT_EphemeralPhysics_v4')


# stream PhysicsHLTPhysics2

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHLTPhysics2_datasetEphemeralHLTPhysics4_selector
streamPhysicsHLTPhysics2_datasetEphemeralHLTPhysics4_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHLTPhysics2_datasetEphemeralHLTPhysics4_selector.l1tResults = cms.InputTag('')
streamPhysicsHLTPhysics2_datasetEphemeralHLTPhysics4_selector.throw      = cms.bool(False)
streamPhysicsHLTPhysics2_datasetEphemeralHLTPhysics4_selector.triggerConditions = cms.vstring('HLT_EphemeralPhysics_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHLTPhysics2_datasetEphemeralHLTPhysics5_selector
streamPhysicsHLTPhysics2_datasetEphemeralHLTPhysics5_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHLTPhysics2_datasetEphemeralHLTPhysics5_selector.l1tResults = cms.InputTag('')
streamPhysicsHLTPhysics2_datasetEphemeralHLTPhysics5_selector.throw      = cms.bool(False)
streamPhysicsHLTPhysics2_datasetEphemeralHLTPhysics5_selector.triggerConditions = cms.vstring('HLT_EphemeralPhysics_v4')


# stream PhysicsHLTPhysics3

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHLTPhysics3_datasetEphemeralHLTPhysics6_selector
streamPhysicsHLTPhysics3_datasetEphemeralHLTPhysics6_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHLTPhysics3_datasetEphemeralHLTPhysics6_selector.l1tResults = cms.InputTag('')
streamPhysicsHLTPhysics3_datasetEphemeralHLTPhysics6_selector.throw      = cms.bool(False)
streamPhysicsHLTPhysics3_datasetEphemeralHLTPhysics6_selector.triggerConditions = cms.vstring('HLT_EphemeralPhysics_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHLTPhysics3_datasetEphemeralHLTPhysics7_selector
streamPhysicsHLTPhysics3_datasetEphemeralHLTPhysics7_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHLTPhysics3_datasetEphemeralHLTPhysics7_selector.l1tResults = cms.InputTag('')
streamPhysicsHLTPhysics3_datasetEphemeralHLTPhysics7_selector.throw      = cms.bool(False)
streamPhysicsHLTPhysics3_datasetEphemeralHLTPhysics7_selector.triggerConditions = cms.vstring('HLT_EphemeralPhysics_v4')


# stream PhysicsJetMET0

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsJetMET0_datasetJetMET0_selector
streamPhysicsJetMET0_datasetJetMET0_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsJetMET0_datasetJetMET0_selector.l1tResults = cms.InputTag('')
streamPhysicsJetMET0_datasetJetMET0_selector.throw      = cms.bool(False)
streamPhysicsJetMET0_datasetJetMET0_selector.triggerConditions = cms.vstring(
    'HLT_AK8DiPFJet250_250_MassSD30_v4',
    'HLT_AK8DiPFJet250_250_MassSD50_v4',
    'HLT_AK8DiPFJet260_260_MassSD30_v4',
    'HLT_AK8DiPFJet260_260_MassSD50_v4',
    'HLT_AK8DiPFJet270_270_MassSD30_v4',
    'HLT_AK8DiPFJet280_280_MassSD30_v4',
    'HLT_AK8DiPFJet290_290_MassSD30_v4',
    'HLT_AK8PFJet140_v19',
    'HLT_AK8PFJet200_v19',
    'HLT_AK8PFJet220_SoftDropMass40_PNetBB0p06_DoubleAK4PFJet60_30_PNet2BTagMean0p50_v1',
    'HLT_AK8PFJet220_SoftDropMass40_PNetBB0p06_DoubleAK4PFJet60_30_PNet2BTagMean0p53_v1',
    'HLT_AK8PFJet220_SoftDropMass40_PNetBB0p06_DoubleAK4PFJet60_30_PNet2BTagMean0p55_v1',
    'HLT_AK8PFJet220_SoftDropMass40_PNetBB0p06_DoubleAK4PFJet60_30_PNet2BTagMean0p60_v1',
    'HLT_AK8PFJet220_SoftDropMass40_v5',
    'HLT_AK8PFJet230_SoftDropMass40_PNetBB0p06_v1',
    'HLT_AK8PFJet230_SoftDropMass40_PNetBB0p10_v1',
    'HLT_AK8PFJet230_SoftDropMass40_PNetTauTau0p03_v1',
    'HLT_AK8PFJet230_SoftDropMass40_PNetTauTau0p05_v1',
    'HLT_AK8PFJet230_SoftDropMass40_v5',
    'HLT_AK8PFJet250_SoftDropMass40_PNetBB0p06_v1',
    'HLT_AK8PFJet250_SoftDropMass40_PNetBB0p10_v1',
    'HLT_AK8PFJet250_SoftDropMass40_PNetTauTau0p03_v1',
    'HLT_AK8PFJet250_SoftDropMass40_PNetTauTau0p05_v1',
    'HLT_AK8PFJet260_v20',
    'HLT_AK8PFJet275_SoftDropMass40_PNetBB0p06_v1',
    'HLT_AK8PFJet275_SoftDropMass40_PNetBB0p10_v1',
    'HLT_AK8PFJet275_SoftDropMass40_PNetTauTau0p03_v1',
    'HLT_AK8PFJet275_SoftDropMass40_PNetTauTau0p05_v1',
    'HLT_AK8PFJet320_v20',
    'HLT_AK8PFJet400_MassSD30_v4',
    'HLT_AK8PFJet400_v20',
    'HLT_AK8PFJet40_v20',
    'HLT_AK8PFJet420_MassSD30_v4',
    'HLT_AK8PFJet425_SoftDropMass40_v5',
    'HLT_AK8PFJet450_MassSD30_v4',
    'HLT_AK8PFJet450_SoftDropMass40_v5',
    'HLT_AK8PFJet450_v20',
    'HLT_AK8PFJet470_MassSD30_v4',
    'HLT_AK8PFJet500_MassSD30_v4',
    'HLT_AK8PFJet500_v20',
    'HLT_AK8PFJet550_v15',
    'HLT_AK8PFJet60_v19',
    'HLT_AK8PFJet80_v20',
    'HLT_AK8PFJetFwd140_v18',
    'HLT_AK8PFJetFwd15_v7',
    'HLT_AK8PFJetFwd200_v18',
    'HLT_AK8PFJetFwd25_v7',
    'HLT_AK8PFJetFwd260_v19',
    'HLT_AK8PFJetFwd320_v19',
    'HLT_AK8PFJetFwd400_v19',
    'HLT_AK8PFJetFwd40_v19',
    'HLT_AK8PFJetFwd450_v19',
    'HLT_AK8PFJetFwd500_v19',
    'HLT_AK8PFJetFwd60_v18',
    'HLT_AK8PFJetFwd80_v18',
    'HLT_CaloJet500_NoJetID_v16',
    'HLT_CaloJet550_NoJetID_v11',
    'HLT_CaloMET350_NotCleaned_v8',
    'HLT_CaloMET90_NotCleaned_v8',
    'HLT_CaloMHT90_v8',
    'HLT_DiPFJetAve100_HFJEC_v21',
    'HLT_DiPFJetAve140_v17',
    'HLT_DiPFJetAve160_HFJEC_v20',
    'HLT_DiPFJetAve200_v17',
    'HLT_DiPFJetAve220_HFJEC_v20',
    'HLT_DiPFJetAve260_HFJEC_v3',
    'HLT_DiPFJetAve260_v18',
    'HLT_DiPFJetAve300_HFJEC_v20',
    'HLT_DiPFJetAve320_v18',
    'HLT_DiPFJetAve400_v18',
    'HLT_DiPFJetAve40_v18',
    'HLT_DiPFJetAve500_v18',
    'HLT_DiPFJetAve60_HFJEC_v19',
    'HLT_DiPFJetAve60_v18',
    'HLT_DiPFJetAve80_HFJEC_v21',
    'HLT_DiPFJetAve80_v18',
    'HLT_DoublePFJets100_PFBTagDeepJet_p71_v5',
    'HLT_DoublePFJets116MaxDeta1p6_DoublePFBTagDeepJet_p71_v5',
    'HLT_DoublePFJets128MaxDeta1p6_DoublePFBTagDeepJet_p71_v5',
    'HLT_DoublePFJets200_PFBTagDeepJet_p71_v5',
    'HLT_DoublePFJets350_PFBTagDeepJet_p71_v6',
    'HLT_DoublePFJets40_PFBTagDeepJet_p71_v5',
    'HLT_L1ETMHadSeeds_v5',
    'HLT_MET105_IsoTrk50_v13',
    'HLT_MET120_IsoTrk50_v13',
    'HLT_Mu12_DoublePFJets100_PFBTagDeepJet_p71_v5',
    'HLT_Mu12_DoublePFJets200_PFBTagDeepJet_p71_v5',
    'HLT_Mu12_DoublePFJets350_PFBTagDeepJet_p71_v5',
    'HLT_Mu12_DoublePFJets40MaxDeta1p6_DoublePFBTagDeepJet_p71_v5',
    'HLT_Mu12_DoublePFJets40_PFBTagDeepJet_p71_v5',
    'HLT_Mu12_DoublePFJets54MaxDeta1p6_DoublePFBTagDeepJet_p71_v5',
    'HLT_Mu12eta2p3_PFJet40_v5',
    'HLT_Mu12eta2p3_v5',
    'HLT_PFHT1050_v22',
    'HLT_PFHT180_v21',
    'HLT_PFHT250_v21',
    'HLT_PFHT350_v23',
    'HLT_PFHT370_v21',
    'HLT_PFHT430_v21',
    'HLT_PFHT500_PFMET100_PFMHT100_IDTight_v16',
    'HLT_PFHT500_PFMET110_PFMHT110_IDTight_v16',
    'HLT_PFHT510_v21',
    'HLT_PFHT590_v21',
    'HLT_PFHT680_v21',
    'HLT_PFHT700_PFMET85_PFMHT85_IDTight_v16',
    'HLT_PFHT780_v21',
    'HLT_PFHT800_PFMET75_PFMHT75_IDTight_v16',
    'HLT_PFHT890_v21',
    'HLT_PFJet110_v4',
    'HLT_PFJet140_v23',
    'HLT_PFJet200_v23',
    'HLT_PFJet260_v24',
    'HLT_PFJet320_v24',
    'HLT_PFJet400_v24',
    'HLT_PFJet40_v25',
    'HLT_PFJet450_v25',
    'HLT_PFJet500_v25',
    'HLT_PFJet550_v15',
    'HLT_PFJet60_v25',
    'HLT_PFJet80_v25',
    'HLT_PFJetFwd140_v22',
    'HLT_PFJetFwd200_v22',
    'HLT_PFJetFwd260_v23',
    'HLT_PFJetFwd320_v23',
    'HLT_PFJetFwd400_v23',
    'HLT_PFJetFwd40_v23',
    'HLT_PFJetFwd450_v23',
    'HLT_PFJetFwd500_v23',
    'HLT_PFJetFwd60_v23',
    'HLT_PFJetFwd80_v22',
    'HLT_PFMET105_IsoTrk50_v5',
    'HLT_PFMET120_PFMHT120_IDTight_PFHT60_v13',
    'HLT_PFMET120_PFMHT120_IDTight_v24',
    'HLT_PFMET130_PFMHT130_IDTight_v24',
    'HLT_PFMET140_PFMHT140_IDTight_v24',
    'HLT_PFMET200_BeamHaloCleaned_v13',
    'HLT_PFMET200_NotCleaned_v13',
    'HLT_PFMET250_NotCleaned_v13',
    'HLT_PFMET300_NotCleaned_v13',
    'HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF_v4',
    'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_FilterHF_v4',
    'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v13',
    'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v24',
    'HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_FilterHF_v4',
    'HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v23',
    'HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_FilterHF_v4',
    'HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v23',
    'HLT_PFMETTypeOne140_PFMHT140_IDTight_v15',
    'HLT_PFMETTypeOne200_BeamHaloCleaned_v13',
    'HLT_QuadPFJet100_88_70_30_v2',
    'HLT_QuadPFJet103_88_75_15_v9',
    'HLT_QuadPFJet105_88_75_30_v1',
    'HLT_QuadPFJet105_88_76_15_v9',
    'HLT_QuadPFJet111_90_80_15_v9',
    'HLT_QuadPFJet111_90_80_30_v1'
)


# stream PhysicsJetMET1

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsJetMET1_datasetJetMET1_selector
streamPhysicsJetMET1_datasetJetMET1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsJetMET1_datasetJetMET1_selector.l1tResults = cms.InputTag('')
streamPhysicsJetMET1_datasetJetMET1_selector.throw      = cms.bool(False)
streamPhysicsJetMET1_datasetJetMET1_selector.triggerConditions = cms.vstring(
    'HLT_AK8DiPFJet250_250_MassSD30_v4',
    'HLT_AK8DiPFJet250_250_MassSD50_v4',
    'HLT_AK8DiPFJet260_260_MassSD30_v4',
    'HLT_AK8DiPFJet260_260_MassSD50_v4',
    'HLT_AK8DiPFJet270_270_MassSD30_v4',
    'HLT_AK8DiPFJet280_280_MassSD30_v4',
    'HLT_AK8DiPFJet290_290_MassSD30_v4',
    'HLT_AK8PFJet140_v19',
    'HLT_AK8PFJet200_v19',
    'HLT_AK8PFJet220_SoftDropMass40_PNetBB0p06_DoubleAK4PFJet60_30_PNet2BTagMean0p50_v1',
    'HLT_AK8PFJet220_SoftDropMass40_PNetBB0p06_DoubleAK4PFJet60_30_PNet2BTagMean0p53_v1',
    'HLT_AK8PFJet220_SoftDropMass40_PNetBB0p06_DoubleAK4PFJet60_30_PNet2BTagMean0p55_v1',
    'HLT_AK8PFJet220_SoftDropMass40_PNetBB0p06_DoubleAK4PFJet60_30_PNet2BTagMean0p60_v1',
    'HLT_AK8PFJet220_SoftDropMass40_v5',
    'HLT_AK8PFJet230_SoftDropMass40_PNetBB0p06_v1',
    'HLT_AK8PFJet230_SoftDropMass40_PNetBB0p10_v1',
    'HLT_AK8PFJet230_SoftDropMass40_PNetTauTau0p03_v1',
    'HLT_AK8PFJet230_SoftDropMass40_PNetTauTau0p05_v1',
    'HLT_AK8PFJet230_SoftDropMass40_v5',
    'HLT_AK8PFJet250_SoftDropMass40_PNetBB0p06_v1',
    'HLT_AK8PFJet250_SoftDropMass40_PNetBB0p10_v1',
    'HLT_AK8PFJet250_SoftDropMass40_PNetTauTau0p03_v1',
    'HLT_AK8PFJet250_SoftDropMass40_PNetTauTau0p05_v1',
    'HLT_AK8PFJet260_v20',
    'HLT_AK8PFJet275_SoftDropMass40_PNetBB0p06_v1',
    'HLT_AK8PFJet275_SoftDropMass40_PNetBB0p10_v1',
    'HLT_AK8PFJet275_SoftDropMass40_PNetTauTau0p03_v1',
    'HLT_AK8PFJet275_SoftDropMass40_PNetTauTau0p05_v1',
    'HLT_AK8PFJet320_v20',
    'HLT_AK8PFJet400_MassSD30_v4',
    'HLT_AK8PFJet400_v20',
    'HLT_AK8PFJet40_v20',
    'HLT_AK8PFJet420_MassSD30_v4',
    'HLT_AK8PFJet425_SoftDropMass40_v5',
    'HLT_AK8PFJet450_MassSD30_v4',
    'HLT_AK8PFJet450_SoftDropMass40_v5',
    'HLT_AK8PFJet450_v20',
    'HLT_AK8PFJet470_MassSD30_v4',
    'HLT_AK8PFJet500_MassSD30_v4',
    'HLT_AK8PFJet500_v20',
    'HLT_AK8PFJet550_v15',
    'HLT_AK8PFJet60_v19',
    'HLT_AK8PFJet80_v20',
    'HLT_AK8PFJetFwd140_v18',
    'HLT_AK8PFJetFwd15_v7',
    'HLT_AK8PFJetFwd200_v18',
    'HLT_AK8PFJetFwd25_v7',
    'HLT_AK8PFJetFwd260_v19',
    'HLT_AK8PFJetFwd320_v19',
    'HLT_AK8PFJetFwd400_v19',
    'HLT_AK8PFJetFwd40_v19',
    'HLT_AK8PFJetFwd450_v19',
    'HLT_AK8PFJetFwd500_v19',
    'HLT_AK8PFJetFwd60_v18',
    'HLT_AK8PFJetFwd80_v18',
    'HLT_CaloJet500_NoJetID_v16',
    'HLT_CaloJet550_NoJetID_v11',
    'HLT_CaloMET350_NotCleaned_v8',
    'HLT_CaloMET90_NotCleaned_v8',
    'HLT_CaloMHT90_v8',
    'HLT_DiPFJetAve100_HFJEC_v21',
    'HLT_DiPFJetAve140_v17',
    'HLT_DiPFJetAve160_HFJEC_v20',
    'HLT_DiPFJetAve200_v17',
    'HLT_DiPFJetAve220_HFJEC_v20',
    'HLT_DiPFJetAve260_HFJEC_v3',
    'HLT_DiPFJetAve260_v18',
    'HLT_DiPFJetAve300_HFJEC_v20',
    'HLT_DiPFJetAve320_v18',
    'HLT_DiPFJetAve400_v18',
    'HLT_DiPFJetAve40_v18',
    'HLT_DiPFJetAve500_v18',
    'HLT_DiPFJetAve60_HFJEC_v19',
    'HLT_DiPFJetAve60_v18',
    'HLT_DiPFJetAve80_HFJEC_v21',
    'HLT_DiPFJetAve80_v18',
    'HLT_DoublePFJets100_PFBTagDeepJet_p71_v5',
    'HLT_DoublePFJets116MaxDeta1p6_DoublePFBTagDeepJet_p71_v5',
    'HLT_DoublePFJets128MaxDeta1p6_DoublePFBTagDeepJet_p71_v5',
    'HLT_DoublePFJets200_PFBTagDeepJet_p71_v5',
    'HLT_DoublePFJets350_PFBTagDeepJet_p71_v6',
    'HLT_DoublePFJets40_PFBTagDeepJet_p71_v5',
    'HLT_L1ETMHadSeeds_v5',
    'HLT_MET105_IsoTrk50_v13',
    'HLT_MET120_IsoTrk50_v13',
    'HLT_Mu12_DoublePFJets100_PFBTagDeepJet_p71_v5',
    'HLT_Mu12_DoublePFJets200_PFBTagDeepJet_p71_v5',
    'HLT_Mu12_DoublePFJets350_PFBTagDeepJet_p71_v5',
    'HLT_Mu12_DoublePFJets40MaxDeta1p6_DoublePFBTagDeepJet_p71_v5',
    'HLT_Mu12_DoublePFJets40_PFBTagDeepJet_p71_v5',
    'HLT_Mu12_DoublePFJets54MaxDeta1p6_DoublePFBTagDeepJet_p71_v5',
    'HLT_Mu12eta2p3_PFJet40_v5',
    'HLT_Mu12eta2p3_v5',
    'HLT_PFHT1050_v22',
    'HLT_PFHT180_v21',
    'HLT_PFHT250_v21',
    'HLT_PFHT350_v23',
    'HLT_PFHT370_v21',
    'HLT_PFHT430_v21',
    'HLT_PFHT500_PFMET100_PFMHT100_IDTight_v16',
    'HLT_PFHT500_PFMET110_PFMHT110_IDTight_v16',
    'HLT_PFHT510_v21',
    'HLT_PFHT590_v21',
    'HLT_PFHT680_v21',
    'HLT_PFHT700_PFMET85_PFMHT85_IDTight_v16',
    'HLT_PFHT780_v21',
    'HLT_PFHT800_PFMET75_PFMHT75_IDTight_v16',
    'HLT_PFHT890_v21',
    'HLT_PFJet110_v4',
    'HLT_PFJet140_v23',
    'HLT_PFJet200_v23',
    'HLT_PFJet260_v24',
    'HLT_PFJet320_v24',
    'HLT_PFJet400_v24',
    'HLT_PFJet40_v25',
    'HLT_PFJet450_v25',
    'HLT_PFJet500_v25',
    'HLT_PFJet550_v15',
    'HLT_PFJet60_v25',
    'HLT_PFJet80_v25',
    'HLT_PFJetFwd140_v22',
    'HLT_PFJetFwd200_v22',
    'HLT_PFJetFwd260_v23',
    'HLT_PFJetFwd320_v23',
    'HLT_PFJetFwd400_v23',
    'HLT_PFJetFwd40_v23',
    'HLT_PFJetFwd450_v23',
    'HLT_PFJetFwd500_v23',
    'HLT_PFJetFwd60_v23',
    'HLT_PFJetFwd80_v22',
    'HLT_PFMET105_IsoTrk50_v5',
    'HLT_PFMET120_PFMHT120_IDTight_PFHT60_v13',
    'HLT_PFMET120_PFMHT120_IDTight_v24',
    'HLT_PFMET130_PFMHT130_IDTight_v24',
    'HLT_PFMET140_PFMHT140_IDTight_v24',
    'HLT_PFMET200_BeamHaloCleaned_v13',
    'HLT_PFMET200_NotCleaned_v13',
    'HLT_PFMET250_NotCleaned_v13',
    'HLT_PFMET300_NotCleaned_v13',
    'HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF_v4',
    'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_FilterHF_v4',
    'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v13',
    'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v24',
    'HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_FilterHF_v4',
    'HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v23',
    'HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_FilterHF_v4',
    'HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v23',
    'HLT_PFMETTypeOne140_PFMHT140_IDTight_v15',
    'HLT_PFMETTypeOne200_BeamHaloCleaned_v13',
    'HLT_QuadPFJet100_88_70_30_v2',
    'HLT_QuadPFJet103_88_75_15_v9',
    'HLT_QuadPFJet105_88_75_30_v1',
    'HLT_QuadPFJet105_88_76_15_v9',
    'HLT_QuadPFJet111_90_80_15_v9',
    'HLT_QuadPFJet111_90_80_30_v1'
)


# stream PhysicsMuon0

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsMuon0_datasetMuon0_selector
streamPhysicsMuon0_datasetMuon0_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsMuon0_datasetMuon0_selector.l1tResults = cms.InputTag('')
streamPhysicsMuon0_datasetMuon0_selector.throw      = cms.bool(False)
streamPhysicsMuon0_datasetMuon0_selector.triggerConditions = cms.vstring(
    'HLT_CascadeMu100_v7',
    'HLT_CscCluster_Loose_v4',
    'HLT_CscCluster_Medium_v4',
    'HLT_CscCluster_Tight_v4',
    'HLT_DoubleCscCluster100_v1',
    'HLT_DoubleCscCluster75_v1',
    'HLT_DoubleL2Mu10NoVtx_2Cha_CosmicSeed_VetoL3Mu0DxyMax1cm_v4',
    'HLT_DoubleL2Mu10NoVtx_2Cha_VetoL3Mu0DxyMax1cm_v5',
    'HLT_DoubleL2Mu12NoVtx_2Cha_CosmicSeed_VetoL3Mu0DxyMax1cm_v4',
    'HLT_DoubleL2Mu12NoVtx_2Cha_VetoL3Mu0DxyMax1cm_v4',
    'HLT_DoubleL2Mu14NoVtx_2Cha_VetoL3Mu0DxyMax1cm_v4',
    'HLT_DoubleL2Mu23NoVtx_2Cha_CosmicSeed_v5',
    'HLT_DoubleL2Mu23NoVtx_2Cha_v5',
    'HLT_DoubleL2Mu25NoVtx_2Cha_CosmicSeed_Eta2p4_v5',
    'HLT_DoubleL2Mu25NoVtx_2Cha_CosmicSeed_v5',
    'HLT_DoubleL2Mu25NoVtx_2Cha_Eta2p4_v5',
    'HLT_DoubleL2Mu25NoVtx_2Cha_v5',
    'HLT_DoubleL2Mu30NoVtx_2Cha_CosmicSeed_Eta2p4_v5',
    'HLT_DoubleL2Mu30NoVtx_2Cha_Eta2p4_v5',
    'HLT_DoubleL2Mu50_v5',
    'HLT_DoubleL2Mu_L3Mu16NoVtx_VetoL3Mu0DxyMax0p1cm_v4',
    'HLT_DoubleL2Mu_L3Mu18NoVtx_VetoL3Mu0DxyMax0p1cm_v4',
    'HLT_DoubleL3Mu16_10NoVtx_DxyMin0p01cm_v5',
    'HLT_DoubleL3Mu18_10NoVtx_DxyMin0p01cm_v4',
    'HLT_DoubleL3Mu20_10NoVtx_DxyMin0p01cm_v4',
    'HLT_DoubleL3dTksMu16_10NoVtx_DxyMin0p01cm_v4',
    'HLT_DoubleMu3_DCA_PFMET50_PFMHT60_v14',
    'HLT_DoubleMu3_DZ_PFMET50_PFMHT60_v14',
    'HLT_DoubleMu3_DZ_PFMET70_PFMHT70_v14',
    'HLT_DoubleMu3_DZ_PFMET90_PFMHT90_v14',
    'HLT_DoubleMu43NoFiltersNoVtx_v8',
    'HLT_DoubleMu48NoFiltersNoVtx_v8',
    'HLT_DoubleMu4_Mass3p8_DZ_PFHT350_v12',
    'HLT_HighPtTkMu100_v6',
    'HLT_IsoMu20_eta2p1_LooseDeepTauPFTauHPS27_eta2p1_CrossL1_v5',
    'HLT_IsoMu20_v19',
    'HLT_IsoMu24_OneProng32_v1',
    'HLT_IsoMu24_TwoProngs35_v5',
    'HLT_IsoMu24_eta2p1_LooseDeepTauPFTauHPS180_eta2p1_v5',
    'HLT_IsoMu24_eta2p1_LooseDeepTauPFTauHPS30_eta2p1_CrossL1_v5',
    'HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS20_eta2p1_SingleL1_v4',
    'HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1_v4',
    'HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60_CrossL1_v4',
    'HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet75_CrossL1_v4',
    'HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1_v5',
    'HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS45_L2NN_eta2p1_CrossL1_v4',
    'HLT_IsoMu24_eta2p1_v19',
    'HLT_IsoMu24_v17',
    'HLT_IsoMu27_MediumDeepTauPFTauHPS20_eta2p1_SingleL1_v4',
    'HLT_IsoMu27_v20',
    'HLT_IsoMu50_AK8PFJet220_SoftDropMass40_PNetBB0p06_v1',
    'HLT_IsoMu50_AK8PFJet220_SoftDropMass40_v4',
    'HLT_IsoMu50_AK8PFJet230_SoftDropMass40_PNetBB0p06_v1',
    'HLT_IsoMu50_AK8PFJet230_SoftDropMass40_PNetBB0p10_v1',
    'HLT_IsoMu50_AK8PFJet230_SoftDropMass40_v4',
    'HLT_L1CSCShower_DTCluster50_v4',
    'HLT_L1CSCShower_DTCluster75_v4',
    'HLT_L3dTksMu10_NoVtx_DxyMin0p01cm_v4',
    'HLT_Mu15_IsoVVVL_PFHT450_PFMET50_v19',
    'HLT_Mu15_IsoVVVL_PFHT450_v19',
    'HLT_Mu15_IsoVVVL_PFHT600_v23',
    'HLT_Mu15_v7',
    'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8_v9',
    'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8_v9',
    'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v19',
    'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_v18',
    'HLT_Mu17_TrkIsoVVL_v17',
    'HLT_Mu17_v17',
    'HLT_Mu18_Mu9_SameSign_v8',
    'HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ_Mass3p8_v7',
    'HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ_Mass8_v7',
    'HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ_v7',
    'HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_v7',
    'HLT_Mu19_TrkIsoVVL_v8',
    'HLT_Mu19_v8',
    'HLT_Mu20_v16',
    'HLT_Mu27_v17',
    'HLT_Mu37_TkMu27_v9',
    'HLT_Mu3_PFJet40_v20',
    'HLT_Mu3er1p5_PFJet100er2p5_PFMET100_PFMHT100_IDTight_v6',
    'HLT_Mu3er1p5_PFJet100er2p5_PFMET80_PFMHT80_IDTight_v6',
    'HLT_Mu3er1p5_PFJet100er2p5_PFMET90_PFMHT90_IDTight_v6',
    'HLT_Mu3er1p5_PFJet100er2p5_PFMETNoMu100_PFMHTNoMu100_IDTight_v6',
    'HLT_Mu3er1p5_PFJet100er2p5_PFMETNoMu80_PFMHTNoMu80_IDTight_v6',
    'HLT_Mu3er1p5_PFJet100er2p5_PFMETNoMu90_PFMHTNoMu90_IDTight_v6',
    'HLT_Mu50_IsoVVVL_PFHT450_v19',
    'HLT_Mu50_L1SingleMuShower_v3',
    'HLT_Mu50_v17',
    'HLT_Mu55_v7',
    'HLT_Mu8_TrkIsoVVL_v16',
    'HLT_Mu8_v16',
    'HLT_TripleMu_10_5_5_DZ_v14',
    'HLT_TripleMu_12_10_5_v14',
    'HLT_TripleMu_5_3_3_Mass3p8_DCA_v7',
    'HLT_TripleMu_5_3_3_Mass3p8_DZ_v12',
    'HLT_TrkMu12_DoubleTrkMu5NoFiltersNoVtx_v10'
)


# stream PhysicsMuon1

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsMuon1_datasetMuon1_selector
streamPhysicsMuon1_datasetMuon1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsMuon1_datasetMuon1_selector.l1tResults = cms.InputTag('')
streamPhysicsMuon1_datasetMuon1_selector.throw      = cms.bool(False)
streamPhysicsMuon1_datasetMuon1_selector.triggerConditions = cms.vstring(
    'HLT_CascadeMu100_v7',
    'HLT_CscCluster_Loose_v4',
    'HLT_CscCluster_Medium_v4',
    'HLT_CscCluster_Tight_v4',
    'HLT_DoubleCscCluster100_v1',
    'HLT_DoubleCscCluster75_v1',
    'HLT_DoubleL2Mu10NoVtx_2Cha_CosmicSeed_VetoL3Mu0DxyMax1cm_v4',
    'HLT_DoubleL2Mu10NoVtx_2Cha_VetoL3Mu0DxyMax1cm_v5',
    'HLT_DoubleL2Mu12NoVtx_2Cha_CosmicSeed_VetoL3Mu0DxyMax1cm_v4',
    'HLT_DoubleL2Mu12NoVtx_2Cha_VetoL3Mu0DxyMax1cm_v4',
    'HLT_DoubleL2Mu14NoVtx_2Cha_VetoL3Mu0DxyMax1cm_v4',
    'HLT_DoubleL2Mu23NoVtx_2Cha_CosmicSeed_v5',
    'HLT_DoubleL2Mu23NoVtx_2Cha_v5',
    'HLT_DoubleL2Mu25NoVtx_2Cha_CosmicSeed_Eta2p4_v5',
    'HLT_DoubleL2Mu25NoVtx_2Cha_CosmicSeed_v5',
    'HLT_DoubleL2Mu25NoVtx_2Cha_Eta2p4_v5',
    'HLT_DoubleL2Mu25NoVtx_2Cha_v5',
    'HLT_DoubleL2Mu30NoVtx_2Cha_CosmicSeed_Eta2p4_v5',
    'HLT_DoubleL2Mu30NoVtx_2Cha_Eta2p4_v5',
    'HLT_DoubleL2Mu50_v5',
    'HLT_DoubleL2Mu_L3Mu16NoVtx_VetoL3Mu0DxyMax0p1cm_v4',
    'HLT_DoubleL2Mu_L3Mu18NoVtx_VetoL3Mu0DxyMax0p1cm_v4',
    'HLT_DoubleL3Mu16_10NoVtx_DxyMin0p01cm_v5',
    'HLT_DoubleL3Mu18_10NoVtx_DxyMin0p01cm_v4',
    'HLT_DoubleL3Mu20_10NoVtx_DxyMin0p01cm_v4',
    'HLT_DoubleL3dTksMu16_10NoVtx_DxyMin0p01cm_v4',
    'HLT_DoubleMu3_DCA_PFMET50_PFMHT60_v14',
    'HLT_DoubleMu3_DZ_PFMET50_PFMHT60_v14',
    'HLT_DoubleMu3_DZ_PFMET70_PFMHT70_v14',
    'HLT_DoubleMu3_DZ_PFMET90_PFMHT90_v14',
    'HLT_DoubleMu43NoFiltersNoVtx_v8',
    'HLT_DoubleMu48NoFiltersNoVtx_v8',
    'HLT_DoubleMu4_Mass3p8_DZ_PFHT350_v12',
    'HLT_HighPtTkMu100_v6',
    'HLT_IsoMu20_eta2p1_LooseDeepTauPFTauHPS27_eta2p1_CrossL1_v5',
    'HLT_IsoMu20_v19',
    'HLT_IsoMu24_OneProng32_v1',
    'HLT_IsoMu24_TwoProngs35_v5',
    'HLT_IsoMu24_eta2p1_LooseDeepTauPFTauHPS180_eta2p1_v5',
    'HLT_IsoMu24_eta2p1_LooseDeepTauPFTauHPS30_eta2p1_CrossL1_v5',
    'HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS20_eta2p1_SingleL1_v4',
    'HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1_v4',
    'HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60_CrossL1_v4',
    'HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet75_CrossL1_v4',
    'HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1_v5',
    'HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS45_L2NN_eta2p1_CrossL1_v4',
    'HLT_IsoMu24_eta2p1_v19',
    'HLT_IsoMu24_v17',
    'HLT_IsoMu27_MediumDeepTauPFTauHPS20_eta2p1_SingleL1_v4',
    'HLT_IsoMu27_v20',
    'HLT_IsoMu50_AK8PFJet220_SoftDropMass40_PNetBB0p06_v1',
    'HLT_IsoMu50_AK8PFJet220_SoftDropMass40_v4',
    'HLT_IsoMu50_AK8PFJet230_SoftDropMass40_PNetBB0p06_v1',
    'HLT_IsoMu50_AK8PFJet230_SoftDropMass40_PNetBB0p10_v1',
    'HLT_IsoMu50_AK8PFJet230_SoftDropMass40_v4',
    'HLT_L1CSCShower_DTCluster50_v4',
    'HLT_L1CSCShower_DTCluster75_v4',
    'HLT_L3dTksMu10_NoVtx_DxyMin0p01cm_v4',
    'HLT_Mu15_IsoVVVL_PFHT450_PFMET50_v19',
    'HLT_Mu15_IsoVVVL_PFHT450_v19',
    'HLT_Mu15_IsoVVVL_PFHT600_v23',
    'HLT_Mu15_v7',
    'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8_v9',
    'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8_v9',
    'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v19',
    'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_v18',
    'HLT_Mu17_TrkIsoVVL_v17',
    'HLT_Mu17_v17',
    'HLT_Mu18_Mu9_SameSign_v8',
    'HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ_Mass3p8_v7',
    'HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ_Mass8_v7',
    'HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ_v7',
    'HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_v7',
    'HLT_Mu19_TrkIsoVVL_v8',
    'HLT_Mu19_v8',
    'HLT_Mu20_v16',
    'HLT_Mu27_v17',
    'HLT_Mu37_TkMu27_v9',
    'HLT_Mu3_PFJet40_v20',
    'HLT_Mu3er1p5_PFJet100er2p5_PFMET100_PFMHT100_IDTight_v6',
    'HLT_Mu3er1p5_PFJet100er2p5_PFMET80_PFMHT80_IDTight_v6',
    'HLT_Mu3er1p5_PFJet100er2p5_PFMET90_PFMHT90_IDTight_v6',
    'HLT_Mu3er1p5_PFJet100er2p5_PFMETNoMu100_PFMHTNoMu100_IDTight_v6',
    'HLT_Mu3er1p5_PFJet100er2p5_PFMETNoMu80_PFMHTNoMu80_IDTight_v6',
    'HLT_Mu3er1p5_PFJet100er2p5_PFMETNoMu90_PFMHTNoMu90_IDTight_v6',
    'HLT_Mu50_IsoVVVL_PFHT450_v19',
    'HLT_Mu50_L1SingleMuShower_v3',
    'HLT_Mu50_v17',
    'HLT_Mu55_v7',
    'HLT_Mu8_TrkIsoVVL_v16',
    'HLT_Mu8_v16',
    'HLT_TripleMu_10_5_5_DZ_v14',
    'HLT_TripleMu_12_10_5_v14',
    'HLT_TripleMu_5_3_3_Mass3p8_DCA_v7',
    'HLT_TripleMu_5_3_3_Mass3p8_DZ_v12',
    'HLT_TrkMu12_DoubleTrkMu5NoFiltersNoVtx_v10'
)


# stream PhysicsReservedDoubleMuonLowMass

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsReservedDoubleMuonLowMass_datasetReservedDoubleMuonLowMass_selector
streamPhysicsReservedDoubleMuonLowMass_datasetReservedDoubleMuonLowMass_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsReservedDoubleMuonLowMass_datasetReservedDoubleMuonLowMass_selector.l1tResults = cms.InputTag('')
streamPhysicsReservedDoubleMuonLowMass_datasetReservedDoubleMuonLowMass_selector.throw      = cms.bool(False)
streamPhysicsReservedDoubleMuonLowMass_datasetReservedDoubleMuonLowMass_selector.triggerConditions = cms.vstring(
    'HLT_Dimuon0_Jpsi3p5_Muon2_v9',
    'HLT_Dimuon0_Jpsi_L1_4R_0er1p5R_v11',
    'HLT_Dimuon0_Jpsi_L1_NoOS_v11',
    'HLT_Dimuon0_Jpsi_NoVertexing_L1_4R_0er1p5R_v11',
    'HLT_Dimuon0_Jpsi_NoVertexing_NoOS_v11',
    'HLT_Dimuon0_Jpsi_NoVertexing_v12',
    'HLT_Dimuon0_Jpsi_v12',
    'HLT_Dimuon0_LowMass_L1_0er1p5R_v11',
    'HLT_Dimuon0_LowMass_L1_0er1p5_v12',
    'HLT_Dimuon0_LowMass_L1_4R_v11',
    'HLT_Dimuon0_LowMass_L1_4_v12',
    'HLT_Dimuon0_LowMass_L1_TM530_v10',
    'HLT_Dimuon0_LowMass_v12',
    'HLT_Dimuon0_Upsilon_L1_4p5_v13',
    'HLT_Dimuon0_Upsilon_L1_4p5er2p0M_v11',
    'HLT_Dimuon0_Upsilon_L1_4p5er2p0_v13',
    'HLT_Dimuon0_Upsilon_Muon_NoL1Mass_v10',
    'HLT_Dimuon0_Upsilon_NoVertexing_v11',
    'HLT_Dimuon12_Upsilon_y1p4_v6',
    'HLT_Dimuon14_Phi_Barrel_Seagulls_v11',
    'HLT_Dimuon18_PsiPrime_noCorrL1_v10',
    'HLT_Dimuon18_PsiPrime_v18',
    'HLT_Dimuon24_Phi_noCorrL1_v10',
    'HLT_Dimuon24_Upsilon_noCorrL1_v10',
    'HLT_Dimuon25_Jpsi_noCorrL1_v10',
    'HLT_Dimuon25_Jpsi_v18',
    'HLT_DoubleMu2_Jpsi_DoubleTrk1_Phi1p05_v10',
    'HLT_DoubleMu3_DoubleEle7p5_CaloIdL_TrackIdL_Upsilon_v8',
    'HLT_DoubleMu3_TkMu_DsTau3Mu_v8',
    'HLT_DoubleMu3_Trk_Tau3mu_NoL1Mass_v10',
    'HLT_DoubleMu3_Trk_Tau3mu_v16',
    'HLT_DoubleMu4_3_Bs_v19',
    'HLT_DoubleMu4_3_Jpsi_v19',
    'HLT_DoubleMu4_JpsiTrkTrk_Displaced_v11',
    'HLT_DoubleMu4_Jpsi_Displaced_v11',
    'HLT_DoubleMu4_Jpsi_NoVertexing_v11',
    'HLT_DoubleMu4_MuMuTrk_Displaced_v19',
    'HLT_DoubleMu5_Upsilon_DoubleEle3_CaloIdL_TrackIdL_v8',
    'HLT_Mu25_TkMu0_Phi_v12',
    'HLT_Mu30_TkMu0_Psi_v5',
    'HLT_Mu30_TkMu0_Upsilon_v5',
    'HLT_Mu4_L1DoubleMu_v5',
    'HLT_Mu7p5_L2Mu2_Jpsi_v14',
    'HLT_Mu7p5_L2Mu2_Upsilon_v14',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_Charge1_v8',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_v8',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15_Charge1_v8',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15_v8',
    'HLT_Trimuon5_3p5_2_Upsilon_Muon_v9',
    'HLT_TrimuonOpen_5_3p5_2_Upsilon_Muon_v7'
)


# stream PhysicsScoutingPFMonitor

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsScoutingPFMonitor_datasetScoutingPFMonitor_selector
streamPhysicsScoutingPFMonitor_datasetScoutingPFMonitor_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsScoutingPFMonitor_datasetScoutingPFMonitor_selector.l1tResults = cms.InputTag('')
streamPhysicsScoutingPFMonitor_datasetScoutingPFMonitor_selector.throw      = cms.bool(False)
streamPhysicsScoutingPFMonitor_datasetScoutingPFMonitor_selector.triggerConditions = cms.vstring(
    'DST_Run3_DoubleMu3_PFScoutingPixelTracking_v20',
    'DST_Run3_EG16_EG12_PFScoutingPixelTracking_v20',
    'DST_Run3_EG30_PFScoutingPixelTracking_v20',
    'DST_Run3_JetHT_PFScoutingPixelTracking_v20',
    'HLT_Ele115_CaloIdVT_GsfTrkIdT_v19',
    'HLT_Ele35_WPTight_Gsf_v13',
    'HLT_IsoMu27_v20',
    'HLT_Mu50_v17',
    'HLT_PFHT1050_v22',
    'HLT_Photon200_v18'
)


# stream PhysicsZeroBias0

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsZeroBias0_datasetEphemeralZeroBias0_selector
streamPhysicsZeroBias0_datasetEphemeralZeroBias0_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsZeroBias0_datasetEphemeralZeroBias0_selector.l1tResults = cms.InputTag('')
streamPhysicsZeroBias0_datasetEphemeralZeroBias0_selector.throw      = cms.bool(False)
streamPhysicsZeroBias0_datasetEphemeralZeroBias0_selector.triggerConditions = cms.vstring('HLT_EphemeralZeroBias_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsZeroBias0_datasetEphemeralZeroBias1_selector
streamPhysicsZeroBias0_datasetEphemeralZeroBias1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsZeroBias0_datasetEphemeralZeroBias1_selector.l1tResults = cms.InputTag('')
streamPhysicsZeroBias0_datasetEphemeralZeroBias1_selector.throw      = cms.bool(False)
streamPhysicsZeroBias0_datasetEphemeralZeroBias1_selector.triggerConditions = cms.vstring('HLT_EphemeralZeroBias_v4')


# stream PhysicsZeroBias1

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsZeroBias1_datasetEphemeralZeroBias2_selector
streamPhysicsZeroBias1_datasetEphemeralZeroBias2_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsZeroBias1_datasetEphemeralZeroBias2_selector.l1tResults = cms.InputTag('')
streamPhysicsZeroBias1_datasetEphemeralZeroBias2_selector.throw      = cms.bool(False)
streamPhysicsZeroBias1_datasetEphemeralZeroBias2_selector.triggerConditions = cms.vstring('HLT_EphemeralZeroBias_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsZeroBias1_datasetEphemeralZeroBias3_selector
streamPhysicsZeroBias1_datasetEphemeralZeroBias3_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsZeroBias1_datasetEphemeralZeroBias3_selector.l1tResults = cms.InputTag('')
streamPhysicsZeroBias1_datasetEphemeralZeroBias3_selector.throw      = cms.bool(False)
streamPhysicsZeroBias1_datasetEphemeralZeroBias3_selector.triggerConditions = cms.vstring('HLT_EphemeralZeroBias_v4')


# stream PhysicsZeroBias2

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsZeroBias2_datasetEphemeralZeroBias4_selector
streamPhysicsZeroBias2_datasetEphemeralZeroBias4_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsZeroBias2_datasetEphemeralZeroBias4_selector.l1tResults = cms.InputTag('')
streamPhysicsZeroBias2_datasetEphemeralZeroBias4_selector.throw      = cms.bool(False)
streamPhysicsZeroBias2_datasetEphemeralZeroBias4_selector.triggerConditions = cms.vstring('HLT_EphemeralZeroBias_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsZeroBias2_datasetEphemeralZeroBias5_selector
streamPhysicsZeroBias2_datasetEphemeralZeroBias5_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsZeroBias2_datasetEphemeralZeroBias5_selector.l1tResults = cms.InputTag('')
streamPhysicsZeroBias2_datasetEphemeralZeroBias5_selector.throw      = cms.bool(False)
streamPhysicsZeroBias2_datasetEphemeralZeroBias5_selector.triggerConditions = cms.vstring('HLT_EphemeralZeroBias_v4')


# stream PhysicsZeroBias3

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsZeroBias3_datasetEphemeralZeroBias6_selector
streamPhysicsZeroBias3_datasetEphemeralZeroBias6_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsZeroBias3_datasetEphemeralZeroBias6_selector.l1tResults = cms.InputTag('')
streamPhysicsZeroBias3_datasetEphemeralZeroBias6_selector.throw      = cms.bool(False)
streamPhysicsZeroBias3_datasetEphemeralZeroBias6_selector.triggerConditions = cms.vstring('HLT_EphemeralZeroBias_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsZeroBias3_datasetEphemeralZeroBias7_selector
streamPhysicsZeroBias3_datasetEphemeralZeroBias7_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsZeroBias3_datasetEphemeralZeroBias7_selector.l1tResults = cms.InputTag('')
streamPhysicsZeroBias3_datasetEphemeralZeroBias7_selector.throw      = cms.bool(False)
streamPhysicsZeroBias3_datasetEphemeralZeroBias7_selector.triggerConditions = cms.vstring('HLT_EphemeralZeroBias_v4')

