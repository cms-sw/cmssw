import FWCore.ParameterSet.Config as cms

#Trigger bit requirement
import HLTrigger.HLTfilters.hltHighLevel_cfi as hlt
ReserveDMu = hlt.hltHighLevel.clone()
ReserveDMu.TriggerResultsTag = cms.InputTag( "TriggerResults", "", "HLT" )
ReserveDMu.HLTPaths = cms.vstring(
    'HLT_Dimuon0_Jpsi3p5_Muon2_v*',
    'HLT_Dimuon0_Jpsi_L1_4R_0er1p5R_v*',
    'HLT_Dimuon0_Jpsi_L1_NoOS_v*',
    'HLT_Dimuon0_Jpsi_NoVertexing_L1_4R_0er1p5R_v*',
    'HLT_Dimuon0_Jpsi_NoVertexing_NoOS_v*',
    'HLT_Dimuon0_Jpsi_NoVertexing_v*',
    'HLT_Dimuon0_Jpsi_v*',
    'HLT_Dimuon0_LowMass_L1_0er1p5R_v*',
    'HLT_Dimuon0_LowMass_L1_0er1p5_v*',
    'HLT_Dimuon0_LowMass_L1_4R_v*',
    'HLT_Dimuon0_LowMass_L1_4_v*',
    'HLT_Dimuon0_LowMass_L1_TM530_v*',
    'HLT_Dimuon0_LowMass_v*',
    'HLT_Dimuon0_Upsilon_L1_4p5_v*',
    'HLT_Dimuon0_Upsilon_L1_4p5er2p0M_v*',
    'HLT_Dimuon0_Upsilon_L1_4p5er2p0_v*',
    'HLT_Dimuon0_Upsilon_Muon_NoL1Mass_v*',
    'HLT_Dimuon0_Upsilon_NoVertexing_v*',
    'HLT_Dimuon12_Upsilon_y1p4_v*',
    'HLT_Dimuon14_Phi_Barrel_Seagulls_v*',
    'HLT_Dimuon18_PsiPrime_noCorrL1_v*',
    'HLT_Dimuon18_PsiPrime_v*',
    'HLT_Dimuon24_Phi_noCorrL1_v*',
    'HLT_Dimuon24_Upsilon_noCorrL1_v*',
    'HLT_Dimuon25_Jpsi_noCorrL1_v*',
    'HLT_Dimuon25_Jpsi_v*',
    'HLT_DoubleMu2_Jpsi_DoubleTrk1_Phi1p05_v*',
    'HLT_DoubleMu3_DoubleEle7p5_CaloIdL_TrackIdL_Upsilon_v*',
    'HLT_DoubleMu3_TkMu_DsTau3Mu_v*',
    'HLT_DoubleMu3_Trk_Tau3mu_NoL1Mass_v*',
    'HLT_DoubleMu3_Trk_Tau3mu_v*',
    'HLT_DoubleMu4_3_Bs_v*',
    'HLT_DoubleMu4_3_Jpsi_v*',
    'HLT_DoubleMu4_JpsiTrkTrk_Displaced_v*',
    'HLT_DoubleMu4_Jpsi_Displaced_v*',
    'HLT_DoubleMu4_Jpsi_NoVertexing_v*',
    'HLT_DoubleMu4_MuMuTrk_Displaced_v*',
    'HLT_DoubleMu5_Upsilon_DoubleEle3_CaloIdL_TrackIdL_v*',
    'HLT_Mu25_TkMu0_Phi_v*',
    'HLT_Mu30_TkMu0_Psi_v*',
    'HLT_Mu30_TkMu0_Upsilon_v*',
    'HLT_Mu4_L1DoubleMu_v*',
    'HLT_Mu7p5_L2Mu2_Jpsi_v*',
    'HLT_Mu7p5_L2Mu2_Upsilon_v*',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_Charge1_v*',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_v*',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15_Charge1_v*',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15_v*',
    'HLT_Trimuon5_3p5_2_Upsilon_Muon_v*',
    'HLT_TrimuonOpen_5_3p5_2_Upsilon_Muon_v*')
ReserveDMu.andOr = cms.bool( True )
# we want to intentionally throw and exception
# in case it does not match one of the HLT Paths
ReserveDMu.throw = cms.bool( True )
