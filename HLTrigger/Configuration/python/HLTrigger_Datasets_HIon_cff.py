# /dev/CMSSW_12_4_0/HIon

import FWCore.ParameterSet.Config as cms


# stream PhysicsHICommissioning

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHICommissioning_datasetHIEmptyBX_selector
streamPhysicsHICommissioning_datasetHIEmptyBX_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHICommissioning_datasetHIEmptyBX_selector.l1tResults = cms.InputTag('')
streamPhysicsHICommissioning_datasetHIEmptyBX_selector.throw      = cms.bool(False)
streamPhysicsHICommissioning_datasetHIEmptyBX_selector.triggerConditions = cms.vstring(
    'HLT_HIL1NotBptxOR_v2',
    'HLT_HIL1UnpairedBunchBptxMinus_v2',
    'HLT_HIL1UnpairedBunchBptxPlus_v2'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHICommissioning_datasetHIHLTPhysics_selector
streamPhysicsHICommissioning_datasetHIHLTPhysics_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHICommissioning_datasetHIHLTPhysics_selector.l1tResults = cms.InputTag('')
streamPhysicsHICommissioning_datasetHIHLTPhysics_selector.throw      = cms.bool(False)
streamPhysicsHICommissioning_datasetHIHLTPhysics_selector.triggerConditions = cms.vstring('HLT_HIPhysics_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHICommissioning_datasetHIHcalNZS_selector
streamPhysicsHICommissioning_datasetHIHcalNZS_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHICommissioning_datasetHIHcalNZS_selector.l1tResults = cms.InputTag('')
streamPhysicsHICommissioning_datasetHIHcalNZS_selector.throw      = cms.bool(False)
streamPhysicsHICommissioning_datasetHIHcalNZS_selector.triggerConditions = cms.vstring(
    'HLT_HIHcalNZS_v2',
    'HLT_HIHcalPhiSym_v2'
)


# stream PhysicsHIDoubleMuon

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIDoubleMuon_datasetHIDoubleMuon_selector
streamPhysicsHIDoubleMuon_datasetHIDoubleMuon_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIDoubleMuon_datasetHIDoubleMuon_selector.l1tResults = cms.InputTag('')
streamPhysicsHIDoubleMuon_datasetHIDoubleMuon_selector.throw      = cms.bool(False)
streamPhysicsHIDoubleMuon_datasetHIDoubleMuon_selector.triggerConditions = cms.vstring(
    'HLT_HIL1DoubleMu0_v2',
    'HLT_HIL1DoubleMu10_v2',
    'HLT_HIL1DoubleMuOpen_OS_er1p6_v2',
    'HLT_HIL1DoubleMuOpen_er1p6_v2',
    'HLT_HIL1DoubleMuOpen_v2',
    'HLT_HIL2DoubleMuOpen_v2',
    'HLT_HIL2_L1DoubleMu10_v2',
    'HLT_HIL3DoubleMuOpen_M60120_v2',
    'HLT_HIL3DoubleMuOpen_Upsi_v2',
    'HLT_HIL3DoubleMuOpen_v2',
    'HLT_HIL3Mu0_L2Mu0_v2',
    'HLT_HIL3Mu2p5NHitQ10_L2Mu2_M7toinf_v2',
    'HLT_HIL3Mu2p5NHitQ10_L2Mu2_v2',
    'HLT_HIL3Mu2p5_L1DoubleMu0_v2',
    'HLT_HIL3Mu3NHitQ10_L1DoubleMuOpen_v2',
    'HLT_HIL3Mu3_L1DoubleMuOpen_OS_v2',
    'HLT_HIL3Mu3_L1TripleMuOpen_v2',
    'HLT_HIL3_L1DoubleMu10_v2'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIDoubleMuon_datasetHIDoubleMuonPsiPeri_selector
streamPhysicsHIDoubleMuon_datasetHIDoubleMuonPsiPeri_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIDoubleMuon_datasetHIDoubleMuonPsiPeri_selector.l1tResults = cms.InputTag('')
streamPhysicsHIDoubleMuon_datasetHIDoubleMuonPsiPeri_selector.throw      = cms.bool(False)
streamPhysicsHIDoubleMuon_datasetHIDoubleMuonPsiPeri_selector.triggerConditions = cms.vstring(
    'HLT_HIL1DoubleMuOpen_Centrality_30_100_v2',
    'HLT_HIL1DoubleMuOpen_Centrality_40_100_v2',
    'HLT_HIL1DoubleMuOpen_Centrality_50_100_v2',
    'HLT_HIL1DoubleMuOpen_OS_Centrality_30_100_v2',
    'HLT_HIL1DoubleMuOpen_OS_Centrality_40_100_v2',
    'HLT_HIL3DoubleMuOpen_JpsiPsi_v2',
    'HLT_HIL3Mu0NHitQ10_L2Mu0_MAXdR3p5_M1to5_v2'
)


# stream PhysicsHIForward

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIForward_datasetHICastor_selector
streamPhysicsHIForward_datasetHICastor_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIForward_datasetHICastor_selector.l1tResults = cms.InputTag('')
streamPhysicsHIForward_datasetHICastor_selector.throw      = cms.bool(False)
streamPhysicsHIForward_datasetHICastor_selector.triggerConditions = cms.vstring(
    'HLT_HICastor_HighJet_BptxAND_v2',
    'HLT_HICastor_HighJet_MBHF1AND_BptxAND_v2',
    'HLT_HICastor_HighJet_MBHF1OR_BptxAND_v2',
    'HLT_HICastor_HighJet_MBHF2AND_BptxAND_v2',
    'HLT_HICastor_HighJet_NotMBHF2AND_v2',
    'HLT_HICastor_HighJet_NotMBHF2OR_v2',
    'HLT_HICastor_HighJet_v2',
    'HLT_HICastor_MediumJet_BptxAND_v2',
    'HLT_HICastor_MediumJet_MBHF1OR_BptxAND_v2',
    'HLT_HICastor_MediumJet_NotMBHF2AND_v2',
    'HLT_HICastor_MediumJet_NotMBHF2OR_v2',
    'HLT_HICastor_MediumJet_SingleEG5_MBHF1OR_BptxAND_v2',
    'HLT_HICastor_MediumJet_SingleMu0_MBHF1OR_BptxAND_v2',
    'HLT_HICastor_MediumJet_v2',
    'HLT_HICastor_Muon_BptxAND_v2',
    'HLT_HICastor_Muon_v2',
    'HLT_HIL1_ZDC_AND_OR_MinimumBiasHF1_AND_BptxAND_v2',
    'HLT_HIL1_ZDC_AND_OR_MinimumBiasHF2_AND_BptxAND_v2'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIForward_datasetHIForward_selector
streamPhysicsHIForward_datasetHIForward_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIForward_datasetHIForward_selector.l1tResults = cms.InputTag('')
streamPhysicsHIForward_datasetHIForward_selector.throw      = cms.bool(False)
streamPhysicsHIForward_datasetHIForward_selector.triggerConditions = cms.vstring(
    'HLT_HIUPC_DoubleEG2_BptxAND_MaxPixelTrack_v2',
    'HLT_HIUPC_DoubleEG2_BptxAND_SinglePixelTrack_MaxPixelTrack_v2',
    'HLT_HIUPC_DoubleEG2_NotMBHF2AND_SinglePixelTrack_MaxPixelTrack_v2',
    'HLT_HIUPC_DoubleEG2_NotMBHF2AND_SinglePixelTrack_v2',
    'HLT_HIUPC_DoubleEG2_NotMBHF2AND_v2',
    'HLT_HIUPC_DoubleEG2_NotMBHF2OR_SinglePixelTrack_MaxPixelTrack_v2',
    'HLT_HIUPC_DoubleEG2_NotMBHF2OR_SinglePixelTrack_v2',
    'HLT_HIUPC_DoubleEG2_NotMBHF2OR_v2',
    'HLT_HIUPC_DoubleEG5_BptxAND_MaxPixelTrack_v2',
    'HLT_HIUPC_DoubleEG5_BptxAND_SinglePixelTrack_MaxPixelTrack_v2',
    'HLT_HIUPC_DoubleEG5_NotMBHF2AND_SinglePixelTrack_MaxPixelTrack_v2',
    'HLT_HIUPC_DoubleEG5_NotMBHF2AND_SinglePixelTrack_v2',
    'HLT_HIUPC_DoubleEG5_NotMBHF2AND_v2',
    'HLT_HIUPC_DoubleEG5_NotMBHF2OR_SinglePixelTrack_MaxPixelTrack_v2',
    'HLT_HIUPC_DoubleEG5_NotMBHF2OR_SinglePixelTrack_v2',
    'HLT_HIUPC_DoubleEG5_NotMBHF2OR_v2',
    'HLT_HIUPC_DoubleMu0_BptxAND_MaxPixelTrack_v2',
    'HLT_HIUPC_DoubleMu0_NotMBHF2AND_MaxPixelTrack_v2',
    'HLT_HIUPC_DoubleMu0_NotMBHF2AND_v2',
    'HLT_HIUPC_DoubleMu0_NotMBHF2OR_MaxPixelTrack_v2',
    'HLT_HIUPC_DoubleMu0_NotMBHF2OR_v2',
    'HLT_HIUPC_DoubleMuOpen_BptxAND_MaxPixelTrack_v2',
    'HLT_HIUPC_DoubleMuOpen_NotMBHF2OR_MaxPixelTrack_v2',
    'HLT_HIUPC_DoubleMuOpen_NotMBHF2OR_v2',
    'HLT_HIUPC_ETT5_Asym50_NotMBHF2OR_SinglePixelTrack_v2',
    'HLT_HIUPC_ETT5_Asym50_NotMBHF2OR_v2',
    'HLT_HIUPC_Mu8_Mu13_MaxPixelTrack_v2',
    'HLT_HIUPC_Mu8_Mu13_v2',
    'HLT_HIUPC_NotMBHF2AND_SinglePixelTrack_MaxPixelTrack_v2',
    'HLT_HIUPC_NotMBHF2AND_SinglePixelTrack_v2',
    'HLT_HIUPC_NotMBHF2AND_v2',
    'HLT_HIUPC_NotMBHF2OR_BptxAND_SinglePixelTrack_MaxPixelTrack_v2',
    'HLT_HIUPC_NotMBHF2OR_BptxAND_SinglePixelTrack_v2',
    'HLT_HIUPC_SingleEG3_BptxAND_MaxPixelTrack_v2',
    'HLT_HIUPC_SingleEG3_BptxAND_SinglePixelTrack_MaxPixelTrack_v2',
    'HLT_HIUPC_SingleEG3_NotMBHF2AND_SinglePixelTrack_MaxPixelTrack_v2',
    'HLT_HIUPC_SingleEG3_NotMBHF2AND_SinglePixelTrack_v2',
    'HLT_HIUPC_SingleEG3_NotMBHF2AND_v2',
    'HLT_HIUPC_SingleEG3_NotMBHF2OR_SinglePixelTrack_MaxPixelTrack_v2',
    'HLT_HIUPC_SingleEG3_NotMBHF2OR_SinglePixelTrack_v2',
    'HLT_HIUPC_SingleEG3_NotMBHF2OR_v2',
    'HLT_HIUPC_SingleEG5_BptxAND_MaxPixelTrack_v2',
    'HLT_HIUPC_SingleEG5_BptxAND_SinglePixelTrack_MaxPixelTrack_v2',
    'HLT_HIUPC_SingleEG5_NotMBHF2AND_SinglePixelTrack_MaxPixelTrack_v2',
    'HLT_HIUPC_SingleEG5_NotMBHF2AND_SinglePixelTrack_v2',
    'HLT_HIUPC_SingleEG5_NotMBHF2AND_v2',
    'HLT_HIUPC_SingleEG5_NotMBHF2OR_SinglePixelTrack_MaxPixelTrack_v2',
    'HLT_HIUPC_SingleEG5_NotMBHF2OR_SinglePixelTrack_v2',
    'HLT_HIUPC_SingleEG5_NotMBHF2OR_v2',
    'HLT_HIUPC_SingleMu0_BptxAND_MaxPixelTrack_v2',
    'HLT_HIUPC_SingleMu0_NotMBHF2AND_MaxPixelTrack_v2',
    'HLT_HIUPC_SingleMu0_NotMBHF2AND_v2',
    'HLT_HIUPC_SingleMu0_NotMBHF2OR_MaxPixelTrack_v2',
    'HLT_HIUPC_SingleMu0_NotMBHF2OR_v2',
    'HLT_HIUPC_SingleMu3_BptxAND_MaxPixelTrack_v2',
    'HLT_HIUPC_SingleMu3_NotMBHF2OR_MaxPixelTrack_v2',
    'HLT_HIUPC_SingleMu3_NotMBHF2OR_v2',
    'HLT_HIUPC_SingleMuOpen_BptxAND_MaxPixelTrack_v2',
    'HLT_HIUPC_SingleMuOpen_NotMBHF2AND_MaxPixelTrack_v2',
    'HLT_HIUPC_SingleMuOpen_NotMBHF2AND_v2',
    'HLT_HIUPC_SingleMuOpen_NotMBHF2OR_MaxPixelTrack_v2',
    'HLT_HIUPC_SingleMuOpen_NotMBHF2OR_v2',
    'HLT_HIUPC_ZeroBias_MaxPixelCluster_v2',
    'HLT_HIUPC_ZeroBias_SinglePixelTrack_v2',
    'HLT_HIZeroBias_v2'
)


# stream PhysicsHIHardProbes

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIHardProbes_datasetHIHardProbes_selector
streamPhysicsHIHardProbes_datasetHIHardProbes_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIHardProbes_datasetHIHardProbes_selector.l1tResults = cms.InputTag('')
streamPhysicsHIHardProbes_datasetHIHardProbes_selector.throw      = cms.bool(False)
streamPhysicsHIHardProbes_datasetHIHardProbes_selector.triggerConditions = cms.vstring(
    'HLT_HICsAK4PFJet100Eta1p5_v2',
    'HLT_HICsAK4PFJet120Eta1p5_v2',
    'HLT_HICsAK4PFJet80Eta1p5_v2',
    'HLT_HIDoubleEle10GsfMass50_v2',
    'HLT_HIDoubleEle10Gsf_v2',
    'HLT_HIDoubleEle15GsfMass50_v2',
    'HLT_HIDoubleEle15Gsf_v2',
    'HLT_HIEle10Gsf_PuAK4CaloJet100Eta2p1_v2',
    'HLT_HIEle10Gsf_PuAK4CaloJet40Eta2p1_v2',
    'HLT_HIEle10Gsf_PuAK4CaloJet60Eta2p1_v2',
    'HLT_HIEle10Gsf_PuAK4CaloJet80Eta2p1_v2',
    'HLT_HIEle10Gsf_v2',
    'HLT_HIEle15Ele10GsfMass50_v2',
    'HLT_HIEle15Ele10Gsf_v2',
    'HLT_HIEle15Gsf_PuAK4CaloJet100Eta2p1_v2',
    'HLT_HIEle15Gsf_PuAK4CaloJet40Eta2p1_v2',
    'HLT_HIEle15Gsf_PuAK4CaloJet60Eta2p1_v2',
    'HLT_HIEle15Gsf_PuAK4CaloJet80Eta2p1_v2',
    'HLT_HIEle15Gsf_v2',
    'HLT_HIEle20Gsf_PuAK4CaloJet100Eta2p1_v2',
    'HLT_HIEle20Gsf_PuAK4CaloJet40Eta2p1_v2',
    'HLT_HIEle20Gsf_PuAK4CaloJet60Eta2p1_v2',
    'HLT_HIEle20Gsf_PuAK4CaloJet80Eta2p1_v2',
    'HLT_HIEle20Gsf_v2',
    'HLT_HIEle30Gsf_v2',
    'HLT_HIEle40Gsf_v2',
    'HLT_HIEle50Gsf_v2',
    'HLT_HIGEDPhoton30_EB_HECut_v2',
    'HLT_HIGEDPhoton30_EB_v2',
    'HLT_HIGEDPhoton30_HECut_v2',
    'HLT_HIGEDPhoton30_v2',
    'HLT_HIGEDPhoton40_EB_HECut_v2',
    'HLT_HIGEDPhoton40_EB_v2',
    'HLT_HIGEDPhoton40_HECut_v2',
    'HLT_HIGEDPhoton40_v2',
    'HLT_HIGEDPhoton50_EB_HECut_v2',
    'HLT_HIGEDPhoton50_EB_v2',
    'HLT_HIGEDPhoton50_HECut_v2',
    'HLT_HIGEDPhoton50_v2',
    'HLT_HIGEDPhoton60_EB_HECut_v2',
    'HLT_HIGEDPhoton60_EB_v2',
    'HLT_HIGEDPhoton60_HECut_v2',
    'HLT_HIGEDPhoton60_v2',
    'HLT_HIIslandPhoton30_Eta1p5_v2',
    'HLT_HIIslandPhoton30_Eta2p4_v2',
    'HLT_HIIslandPhoton40_Eta1p5_v2',
    'HLT_HIIslandPhoton40_Eta2p4_v2',
    'HLT_HIIslandPhoton50_Eta1p5_v2',
    'HLT_HIIslandPhoton50_Eta2p4_v2',
    'HLT_HIIslandPhoton60_Eta1p5_v2',
    'HLT_HIIslandPhoton60_Eta2p4_v2',
    'HLT_HIL1Mu3Eta2p5_Ele10Gsf_v2',
    'HLT_HIL1Mu3Eta2p5_Ele15Gsf_v2',
    'HLT_HIL1Mu3Eta2p5_Ele20Gsf_v2',
    'HLT_HIL1Mu5Eta2p5_Ele10Gsf_v2',
    'HLT_HIL1Mu5Eta2p5_Ele15Gsf_v2',
    'HLT_HIL1Mu5Eta2p5_Ele20Gsf_v2',
    'HLT_HIL1Mu7Eta2p5_Ele10Gsf_v2',
    'HLT_HIL1Mu7Eta2p5_Ele15Gsf_v2',
    'HLT_HIL1Mu7Eta2p5_Ele20Gsf_v2',
    'HLT_HIL3Mu3_EG10HECut_v2',
    'HLT_HIL3Mu3_EG15HECut_v2',
    'HLT_HIL3Mu3_EG20HECut_v2',
    'HLT_HIL3Mu3_EG30HECut_v2',
    'HLT_HIL3Mu5_EG10HECut_v2',
    'HLT_HIL3Mu5_EG15HECut_v2',
    'HLT_HIL3Mu5_EG20HECut_v2',
    'HLT_HIL3Mu5_EG30HECut_v2',
    'HLT_HIL3Mu7_EG10HECut_v2',
    'HLT_HIL3Mu7_EG15HECut_v2',
    'HLT_HIL3Mu7_EG20HECut_v2',
    'HLT_HIL3Mu7_EG30HECut_v2',
    'HLT_HIPuAK4CaloJet100Eta2p4_CSVv2WP0p75_v2',
    'HLT_HIPuAK4CaloJet100Eta2p4_DeepCSV0p4_v2',
    'HLT_HIPuAK4CaloJet100Eta5p1_v2',
    'HLT_HIPuAK4CaloJet100Fwd_v2',
    'HLT_HIPuAK4CaloJet100_35_Eta0p7_v2',
    'HLT_HIPuAK4CaloJet100_35_Eta1p1_v2',
    'HLT_HIPuAK4CaloJet120Eta5p1_v2',
    'HLT_HIPuAK4CaloJet120Fwd_v2',
    'HLT_HIPuAK4CaloJet60Eta2p4_CSVv2WP0p75_v2',
    'HLT_HIPuAK4CaloJet60Eta2p4_DeepCSV0p4_v2',
    'HLT_HIPuAK4CaloJet60Fwd_v2',
    'HLT_HIPuAK4CaloJet80Eta2p4_CSVv2WP0p75_v2',
    'HLT_HIPuAK4CaloJet80Eta2p4_DeepCSV0p4_v2',
    'HLT_HIPuAK4CaloJet80Eta5p1_v2',
    'HLT_HIPuAK4CaloJet80Fwd_v2',
    'HLT_HIPuAK4CaloJet80_35_Eta0p7_v2',
    'HLT_HIPuAK4CaloJet80_35_Eta1p1_v2',
    'HLT_HIPuAK4CaloJet80_45_45_Eta2p1_v2'
)


# stream PhysicsHIHardProbesLower

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIHardProbesLower_datasetHIHardProbesLower_selector
streamPhysicsHIHardProbesLower_datasetHIHardProbesLower_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIHardProbesLower_datasetHIHardProbesLower_selector.l1tResults = cms.InputTag('')
streamPhysicsHIHardProbesLower_datasetHIHardProbesLower_selector.throw      = cms.bool(False)
streamPhysicsHIHardProbesLower_datasetHIHardProbesLower_selector.triggerConditions = cms.vstring(
    'HLT_HICsAK4PFJet60Eta1p5_v2',
    'HLT_HIGEDPhoton10_EB_HECut_v2',
    'HLT_HIGEDPhoton10_EB_v2',
    'HLT_HIGEDPhoton10_HECut_v2',
    'HLT_HIGEDPhoton10_v2',
    'HLT_HIGEDPhoton20_EB_HECut_v2',
    'HLT_HIGEDPhoton20_EB_v2',
    'HLT_HIGEDPhoton20_HECut_v2',
    'HLT_HIGEDPhoton20_v2',
    'HLT_HIIslandPhoton10_Eta1p5_v2',
    'HLT_HIIslandPhoton10_Eta2p4_v2',
    'HLT_HIIslandPhoton20_Eta1p5_v2',
    'HLT_HIIslandPhoton20_Eta2p4_v2',
    'HLT_HIPuAK4CaloJet40Eta5p1_v2',
    'HLT_HIPuAK4CaloJet40Fwd_v2',
    'HLT_HIPuAK4CaloJet60Eta5p1_v2'
)


# stream PhysicsHIHardProbesPeripheral

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIHardProbesPeripheral_datasetHIHardProbesPeripheral_selector
streamPhysicsHIHardProbesPeripheral_datasetHIHardProbesPeripheral_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIHardProbesPeripheral_datasetHIHardProbesPeripheral_selector.l1tResults = cms.InputTag('')
streamPhysicsHIHardProbesPeripheral_datasetHIHardProbesPeripheral_selector.throw      = cms.bool(False)
streamPhysicsHIHardProbesPeripheral_datasetHIHardProbesPeripheral_selector.triggerConditions = cms.vstring(
    'HLT_HICsAK4PFJet100Eta1p5_Centrality_30_100_v2',
    'HLT_HICsAK4PFJet100Eta1p5_Centrality_50_100_v2',
    'HLT_HICsAK4PFJet60Eta1p5_Centrality_30_100_v2',
    'HLT_HICsAK4PFJet60Eta1p5_Centrality_50_100_v2',
    'HLT_HICsAK4PFJet80Eta1p5_Centrality_30_100_v2',
    'HLT_HICsAK4PFJet80Eta1p5_Centrality_50_100_v2',
    'HLT_HIGEDPhoton10_Cent30_100_v2',
    'HLT_HIGEDPhoton10_Cent50_100_v2',
    'HLT_HIGEDPhoton20_Cent30_100_v2',
    'HLT_HIGEDPhoton20_Cent50_100_v2',
    'HLT_HIGEDPhoton30_Cent30_100_v2',
    'HLT_HIGEDPhoton30_Cent50_100_v2',
    'HLT_HIGEDPhoton40_Cent30_100_v2',
    'HLT_HIGEDPhoton40_Cent50_100_v2',
    'HLT_HIIslandPhoton10_Eta2p4_Cent30_100_v2',
    'HLT_HIIslandPhoton10_Eta2p4_Cent50_100_v2',
    'HLT_HIIslandPhoton20_Eta2p4_Cent30_100_v2',
    'HLT_HIIslandPhoton20_Eta2p4_Cent50_100_v2',
    'HLT_HIIslandPhoton30_Eta2p4_Cent30_100_v2',
    'HLT_HIIslandPhoton30_Eta2p4_Cent50_100_v2',
    'HLT_HIIslandPhoton40_Eta2p4_Cent30_100_v2',
    'HLT_HIIslandPhoton40_Eta2p4_Cent50_100_v2',
    'HLT_HIPuAK4CaloJet100Eta5p1_Centrality_30_100_v2',
    'HLT_HIPuAK4CaloJet100Eta5p1_Centrality_50_100_v2',
    'HLT_HIPuAK4CaloJet40Eta5p1_Centrality_30_100_v2',
    'HLT_HIPuAK4CaloJet40Eta5p1_Centrality_50_100_v2',
    'HLT_HIPuAK4CaloJet60Eta5p1_Centrality_30_100_v2',
    'HLT_HIPuAK4CaloJet60Eta5p1_Centrality_50_100_v2',
    'HLT_HIPuAK4CaloJet80Eta5p1_Centrality_30_100_v2',
    'HLT_HIPuAK4CaloJet80Eta5p1_Centrality_50_100_v2'
)


# stream PhysicsHIHeavyFlavor

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIHeavyFlavor_datasetHIHeavyFlavor_selector
streamPhysicsHIHeavyFlavor_datasetHIHeavyFlavor_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIHeavyFlavor_datasetHIHeavyFlavor_selector.l1tResults = cms.InputTag('')
streamPhysicsHIHeavyFlavor_datasetHIHeavyFlavor_selector.throw      = cms.bool(False)
streamPhysicsHIHeavyFlavor_datasetHIHeavyFlavor_selector.triggerConditions = cms.vstring(
    'HLT_HIDmesonPPTrackingGlobal_Dpt15_NoIter10_v2',
    'HLT_HIDmesonPPTrackingGlobal_Dpt15_v2',
    'HLT_HIDmesonPPTrackingGlobal_Dpt20_NoIter10_v2',
    'HLT_HIDmesonPPTrackingGlobal_Dpt20_v2',
    'HLT_HIDmesonPPTrackingGlobal_Dpt30_NoIter10_v2',
    'HLT_HIDmesonPPTrackingGlobal_Dpt30_v2',
    'HLT_HIDmesonPPTrackingGlobal_Dpt40_NoIter10_v2',
    'HLT_HIDmesonPPTrackingGlobal_Dpt40_v2',
    'HLT_HIDmesonPPTrackingGlobal_Dpt50_NoIter10_v2',
    'HLT_HIDmesonPPTrackingGlobal_Dpt50_v2',
    'HLT_HIDmesonPPTrackingGlobal_Dpt60_NoIter10_v2',
    'HLT_HIDmesonPPTrackingGlobal_Dpt60_v2',
    'HLT_HIDsPPTrackingGlobal_Dpt20_NoIter10_v2',
    'HLT_HIDsPPTrackingGlobal_Dpt20_v2',
    'HLT_HIDsPPTrackingGlobal_Dpt30_NoIter10_v2',
    'HLT_HIDsPPTrackingGlobal_Dpt30_v2',
    'HLT_HIDsPPTrackingGlobal_Dpt40_NoIter10_v2',
    'HLT_HIDsPPTrackingGlobal_Dpt40_v2',
    'HLT_HIDsPPTrackingGlobal_Dpt50_NoIter10_v2',
    'HLT_HIDsPPTrackingGlobal_Dpt50_v2',
    'HLT_HIDsPPTrackingGlobal_Dpt60_NoIter10_v2',
    'HLT_HIDsPPTrackingGlobal_Dpt60_v2',
    'HLT_HIFullTracks2018_HighPt18_NoIter10_v2',
    'HLT_HIFullTracks2018_HighPt18_v2',
    'HLT_HIFullTracks2018_HighPt24_NoIter10_v2',
    'HLT_HIFullTracks2018_HighPt24_v2',
    'HLT_HIFullTracks2018_HighPt34_NoIter10_v2',
    'HLT_HIFullTracks2018_HighPt34_v2',
    'HLT_HIFullTracks2018_HighPt45_NoIter10_v2',
    'HLT_HIFullTracks2018_HighPt45_v2',
    'HLT_HIFullTracks2018_HighPt56_NoIter10_v2',
    'HLT_HIFullTracks2018_HighPt56_v2',
    'HLT_HIFullTracks2018_HighPt60_NoIter10_v2',
    'HLT_HIFullTracks2018_HighPt60_v2',
    'HLT_HILcPPTrackingGlobal_Dpt20_NoIter10_v2',
    'HLT_HILcPPTrackingGlobal_Dpt20_v2',
    'HLT_HILcPPTrackingGlobal_Dpt30_NoIter10_v2',
    'HLT_HILcPPTrackingGlobal_Dpt30_v2',
    'HLT_HILcPPTrackingGlobal_Dpt40_NoIter10_v2',
    'HLT_HILcPPTrackingGlobal_Dpt40_v2',
    'HLT_HILcPPTrackingGlobal_Dpt50_NoIter10_v2',
    'HLT_HILcPPTrackingGlobal_Dpt50_v2',
    'HLT_HILcPPTrackingGlobal_Dpt60_NoIter10_v2',
    'HLT_HILcPPTrackingGlobal_Dpt60_v2'
)


# stream PhysicsHIHighMultiplicity

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIHighMultiplicity_datasetHIHighMultiplicityETTAsym_selector
streamPhysicsHIHighMultiplicity_datasetHIHighMultiplicityETTAsym_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIHighMultiplicity_datasetHIHighMultiplicityETTAsym_selector.l1tResults = cms.InputTag('')
streamPhysicsHIHighMultiplicity_datasetHIHighMultiplicityETTAsym_selector.throw      = cms.bool(False)
streamPhysicsHIHighMultiplicity_datasetHIHighMultiplicityETTAsym_selector.triggerConditions = cms.vstring(
    'HLT_HIL1_ETT10_ETTAsym50_MinimumBiasHF1_OR_BptxAND_v2',
    'HLT_HIL1_ETT60_ETTAsym65_MinimumBiasHF2_OR_PixelTracks10_v2',
    'HLT_HIL1_ETT65_ETTAsym80_MinimumBiasHF2_OR_PixelTracks10_v2',
    'HLT_HIL1_ETT8_ETTAsym50_MinimumBiasHF1_OR_BptxAND_v2'
)


# stream PhysicsHILowMultiplicity

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHILowMultiplicity_datasetHILowMultiplicity_selector
streamPhysicsHILowMultiplicity_datasetHILowMultiplicity_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHILowMultiplicity_datasetHILowMultiplicity_selector.l1tResults = cms.InputTag('')
streamPhysicsHILowMultiplicity_datasetHILowMultiplicity_selector.throw      = cms.bool(False)
streamPhysicsHILowMultiplicity_datasetHILowMultiplicity_selector.triggerConditions = cms.vstring(
    'HLT_HIFullTracks_Multiplicity020_HF1AND_v2',
    'HLT_HIFullTracks_Multiplicity020_HF1OR_v2',
    'HLT_HIFullTracks_Multiplicity020_HF2OR_v2',
    'HLT_HIFullTracks_Multiplicity020_v2',
    'HLT_HIFullTracks_Multiplicity2040_HF1AND_v2',
    'HLT_HIFullTracks_Multiplicity2040_HF1OR_v2',
    'HLT_HIFullTracks_Multiplicity2040_HF2OR_v2',
    'HLT_HIFullTracks_Multiplicity2040_v2',
    'HLT_HIFullTracks_Multiplicity335_HF1OR_v2',
    'HLT_HIFullTracks_Multiplicity4060_v2',
    'HLT_HIFullTracks_Multiplicity6080_v2',
    'HLT_HIFullTracks_Multiplicity80100_v2'
)


# stream PhysicsHILowMultiplicityReducedFormat


# stream PhysicsHIMinimumBias0

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIMinimumBias0_datasetHIMinimumBias0_selector
streamPhysicsHIMinimumBias0_datasetHIMinimumBias0_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIMinimumBias0_datasetHIMinimumBias0_selector.l1tResults = cms.InputTag('')
streamPhysicsHIMinimumBias0_datasetHIMinimumBias0_selector.throw      = cms.bool(False)
streamPhysicsHIMinimumBias0_datasetHIMinimumBias0_selector.triggerConditions = cms.vstring(
    'HLT_HIMinimumBias_SinglePixelTrack_NpixBypass_part0_v2',
    'HLT_HIMinimumBias_SinglePixelTrack_NpixGated_part0_v2'
)


# stream PhysicsHIMinimumBias1

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIMinimumBias1_datasetHIMinimumBias1_selector
streamPhysicsHIMinimumBias1_datasetHIMinimumBias1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIMinimumBias1_datasetHIMinimumBias1_selector.l1tResults = cms.InputTag('')
streamPhysicsHIMinimumBias1_datasetHIMinimumBias1_selector.throw      = cms.bool(False)
streamPhysicsHIMinimumBias1_datasetHIMinimumBias1_selector.triggerConditions = cms.vstring(
    'HLT_HIMinimumBias_SinglePixelTrack_NpixBypass_part1_v2',
    'HLT_HIMinimumBias_SinglePixelTrack_NpixGated_part1_v2'
)


# stream PhysicsHIMinimumBias10

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIMinimumBias10_datasetHIMinimumBias10_selector
streamPhysicsHIMinimumBias10_datasetHIMinimumBias10_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIMinimumBias10_datasetHIMinimumBias10_selector.l1tResults = cms.InputTag('')
streamPhysicsHIMinimumBias10_datasetHIMinimumBias10_selector.throw      = cms.bool(False)
streamPhysicsHIMinimumBias10_datasetHIMinimumBias10_selector.triggerConditions = cms.vstring(
    'HLT_HIMinimumBias_SinglePixelTrack_NpixBypass_part10_v2',
    'HLT_HIMinimumBias_SinglePixelTrack_NpixGated_part10_v2'
)


# stream PhysicsHIMinimumBias11

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIMinimumBias11_datasetHIMinimumBias11_selector
streamPhysicsHIMinimumBias11_datasetHIMinimumBias11_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIMinimumBias11_datasetHIMinimumBias11_selector.l1tResults = cms.InputTag('')
streamPhysicsHIMinimumBias11_datasetHIMinimumBias11_selector.throw      = cms.bool(False)
streamPhysicsHIMinimumBias11_datasetHIMinimumBias11_selector.triggerConditions = cms.vstring(
    'HLT_HIMinimumBias_SinglePixelTrack_NpixBypass_part11_v2',
    'HLT_HIMinimumBias_SinglePixelTrack_NpixGated_part11_v2'
)


# stream PhysicsHIMinimumBias12

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIMinimumBias12_datasetHIMinimumBias12_selector
streamPhysicsHIMinimumBias12_datasetHIMinimumBias12_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIMinimumBias12_datasetHIMinimumBias12_selector.l1tResults = cms.InputTag('')
streamPhysicsHIMinimumBias12_datasetHIMinimumBias12_selector.throw      = cms.bool(False)
streamPhysicsHIMinimumBias12_datasetHIMinimumBias12_selector.triggerConditions = cms.vstring(
    'HLT_HIMinimumBias_SinglePixelTrack_NpixBypass_part12_v2',
    'HLT_HIMinimumBias_SinglePixelTrack_NpixGated_part12_v2'
)


# stream PhysicsHIMinimumBias13

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIMinimumBias13_datasetHIMinimumBias13_selector
streamPhysicsHIMinimumBias13_datasetHIMinimumBias13_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIMinimumBias13_datasetHIMinimumBias13_selector.l1tResults = cms.InputTag('')
streamPhysicsHIMinimumBias13_datasetHIMinimumBias13_selector.throw      = cms.bool(False)
streamPhysicsHIMinimumBias13_datasetHIMinimumBias13_selector.triggerConditions = cms.vstring(
    'HLT_HIMinimumBias_SinglePixelTrack_NpixBypass_part13_v2',
    'HLT_HIMinimumBias_SinglePixelTrack_NpixGated_part13_v2'
)


# stream PhysicsHIMinimumBias14

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIMinimumBias14_datasetHIMinimumBias14_selector
streamPhysicsHIMinimumBias14_datasetHIMinimumBias14_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIMinimumBias14_datasetHIMinimumBias14_selector.l1tResults = cms.InputTag('')
streamPhysicsHIMinimumBias14_datasetHIMinimumBias14_selector.throw      = cms.bool(False)
streamPhysicsHIMinimumBias14_datasetHIMinimumBias14_selector.triggerConditions = cms.vstring(
    'HLT_HIMinimumBias_SinglePixelTrack_NpixBypass_part14_v2',
    'HLT_HIMinimumBias_SinglePixelTrack_NpixGated_part14_v2'
)


# stream PhysicsHIMinimumBias15

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIMinimumBias15_datasetHIMinimumBias15_selector
streamPhysicsHIMinimumBias15_datasetHIMinimumBias15_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIMinimumBias15_datasetHIMinimumBias15_selector.l1tResults = cms.InputTag('')
streamPhysicsHIMinimumBias15_datasetHIMinimumBias15_selector.throw      = cms.bool(False)
streamPhysicsHIMinimumBias15_datasetHIMinimumBias15_selector.triggerConditions = cms.vstring(
    'HLT_HIMinimumBias_SinglePixelTrack_NpixBypass_part15_v2',
    'HLT_HIMinimumBias_SinglePixelTrack_NpixGated_part15_v2'
)


# stream PhysicsHIMinimumBias16

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIMinimumBias16_datasetHIMinimumBias16_selector
streamPhysicsHIMinimumBias16_datasetHIMinimumBias16_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIMinimumBias16_datasetHIMinimumBias16_selector.l1tResults = cms.InputTag('')
streamPhysicsHIMinimumBias16_datasetHIMinimumBias16_selector.throw      = cms.bool(False)
streamPhysicsHIMinimumBias16_datasetHIMinimumBias16_selector.triggerConditions = cms.vstring(
    'HLT_HIMinimumBias_SinglePixelTrack_NpixBypass_part16_v2',
    'HLT_HIMinimumBias_SinglePixelTrack_NpixGated_part16_v2'
)


# stream PhysicsHIMinimumBias17

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIMinimumBias17_datasetHIMinimumBias17_selector
streamPhysicsHIMinimumBias17_datasetHIMinimumBias17_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIMinimumBias17_datasetHIMinimumBias17_selector.l1tResults = cms.InputTag('')
streamPhysicsHIMinimumBias17_datasetHIMinimumBias17_selector.throw      = cms.bool(False)
streamPhysicsHIMinimumBias17_datasetHIMinimumBias17_selector.triggerConditions = cms.vstring(
    'HLT_HIMinimumBias_SinglePixelTrack_NpixBypass_part17_v2',
    'HLT_HIMinimumBias_SinglePixelTrack_NpixGated_part17_v2'
)


# stream PhysicsHIMinimumBias18

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIMinimumBias18_datasetHIMinimumBias18_selector
streamPhysicsHIMinimumBias18_datasetHIMinimumBias18_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIMinimumBias18_datasetHIMinimumBias18_selector.l1tResults = cms.InputTag('')
streamPhysicsHIMinimumBias18_datasetHIMinimumBias18_selector.throw      = cms.bool(False)
streamPhysicsHIMinimumBias18_datasetHIMinimumBias18_selector.triggerConditions = cms.vstring(
    'HLT_HIMinimumBias_SinglePixelTrack_NpixBypass_part18_v2',
    'HLT_HIMinimumBias_SinglePixelTrack_NpixGated_part18_v2'
)


# stream PhysicsHIMinimumBias19

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIMinimumBias19_datasetHIMinimumBias19_selector
streamPhysicsHIMinimumBias19_datasetHIMinimumBias19_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIMinimumBias19_datasetHIMinimumBias19_selector.l1tResults = cms.InputTag('')
streamPhysicsHIMinimumBias19_datasetHIMinimumBias19_selector.throw      = cms.bool(False)
streamPhysicsHIMinimumBias19_datasetHIMinimumBias19_selector.triggerConditions = cms.vstring(
    'HLT_HIMinimumBias_SinglePixelTrack_NpixBypass_part19_v2',
    'HLT_HIMinimumBias_SinglePixelTrack_NpixGated_part19_v2'
)


# stream PhysicsHIMinimumBias2

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIMinimumBias2_datasetHIMinimumBias2_selector
streamPhysicsHIMinimumBias2_datasetHIMinimumBias2_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIMinimumBias2_datasetHIMinimumBias2_selector.l1tResults = cms.InputTag('')
streamPhysicsHIMinimumBias2_datasetHIMinimumBias2_selector.throw      = cms.bool(False)
streamPhysicsHIMinimumBias2_datasetHIMinimumBias2_selector.triggerConditions = cms.vstring(
    'HLT_HIMinimumBias_SinglePixelTrack_NpixBypass_part2_v2',
    'HLT_HIMinimumBias_SinglePixelTrack_NpixGated_part2_v2'
)


# stream PhysicsHIMinimumBias3

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIMinimumBias3_datasetHIMinimumBias3_selector
streamPhysicsHIMinimumBias3_datasetHIMinimumBias3_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIMinimumBias3_datasetHIMinimumBias3_selector.l1tResults = cms.InputTag('')
streamPhysicsHIMinimumBias3_datasetHIMinimumBias3_selector.throw      = cms.bool(False)
streamPhysicsHIMinimumBias3_datasetHIMinimumBias3_selector.triggerConditions = cms.vstring(
    'HLT_HIMinimumBias_SinglePixelTrack_NpixBypass_part3_v2',
    'HLT_HIMinimumBias_SinglePixelTrack_NpixGated_part3_v2'
)


# stream PhysicsHIMinimumBias4

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIMinimumBias4_datasetHIMinimumBias4_selector
streamPhysicsHIMinimumBias4_datasetHIMinimumBias4_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIMinimumBias4_datasetHIMinimumBias4_selector.l1tResults = cms.InputTag('')
streamPhysicsHIMinimumBias4_datasetHIMinimumBias4_selector.throw      = cms.bool(False)
streamPhysicsHIMinimumBias4_datasetHIMinimumBias4_selector.triggerConditions = cms.vstring(
    'HLT_HIMinimumBias_SinglePixelTrack_NpixBypass_part4_v2',
    'HLT_HIMinimumBias_SinglePixelTrack_NpixGated_part4_v2'
)


# stream PhysicsHIMinimumBias5

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIMinimumBias5_datasetHIMinimumBias5_selector
streamPhysicsHIMinimumBias5_datasetHIMinimumBias5_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIMinimumBias5_datasetHIMinimumBias5_selector.l1tResults = cms.InputTag('')
streamPhysicsHIMinimumBias5_datasetHIMinimumBias5_selector.throw      = cms.bool(False)
streamPhysicsHIMinimumBias5_datasetHIMinimumBias5_selector.triggerConditions = cms.vstring(
    'HLT_HIMinimumBias_SinglePixelTrack_NpixBypass_part5_v2',
    'HLT_HIMinimumBias_SinglePixelTrack_NpixGated_part5_v2'
)


# stream PhysicsHIMinimumBias6

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIMinimumBias6_datasetHIMinimumBias6_selector
streamPhysicsHIMinimumBias6_datasetHIMinimumBias6_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIMinimumBias6_datasetHIMinimumBias6_selector.l1tResults = cms.InputTag('')
streamPhysicsHIMinimumBias6_datasetHIMinimumBias6_selector.throw      = cms.bool(False)
streamPhysicsHIMinimumBias6_datasetHIMinimumBias6_selector.triggerConditions = cms.vstring(
    'HLT_HIMinimumBias_SinglePixelTrack_NpixBypass_part6_v2',
    'HLT_HIMinimumBias_SinglePixelTrack_NpixGated_part6_v2'
)


# stream PhysicsHIMinimumBias7

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIMinimumBias7_datasetHIMinimumBias7_selector
streamPhysicsHIMinimumBias7_datasetHIMinimumBias7_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIMinimumBias7_datasetHIMinimumBias7_selector.l1tResults = cms.InputTag('')
streamPhysicsHIMinimumBias7_datasetHIMinimumBias7_selector.throw      = cms.bool(False)
streamPhysicsHIMinimumBias7_datasetHIMinimumBias7_selector.triggerConditions = cms.vstring(
    'HLT_HIMinimumBias_SinglePixelTrack_NpixBypass_part7_v2',
    'HLT_HIMinimumBias_SinglePixelTrack_NpixGated_part7_v2'
)


# stream PhysicsHIMinimumBias8

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIMinimumBias8_datasetHIMinimumBias8_selector
streamPhysicsHIMinimumBias8_datasetHIMinimumBias8_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIMinimumBias8_datasetHIMinimumBias8_selector.l1tResults = cms.InputTag('')
streamPhysicsHIMinimumBias8_datasetHIMinimumBias8_selector.throw      = cms.bool(False)
streamPhysicsHIMinimumBias8_datasetHIMinimumBias8_selector.triggerConditions = cms.vstring(
    'HLT_HIMinimumBias_SinglePixelTrack_NpixBypass_part8_v2',
    'HLT_HIMinimumBias_SinglePixelTrack_NpixGated_part8_v2'
)


# stream PhysicsHIMinimumBias9

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIMinimumBias9_datasetHIMinimumBias9_selector
streamPhysicsHIMinimumBias9_datasetHIMinimumBias9_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIMinimumBias9_datasetHIMinimumBias9_selector.l1tResults = cms.InputTag('')
streamPhysicsHIMinimumBias9_datasetHIMinimumBias9_selector.throw      = cms.bool(False)
streamPhysicsHIMinimumBias9_datasetHIMinimumBias9_selector.triggerConditions = cms.vstring(
    'HLT_HIMinimumBias_SinglePixelTrack_NpixBypass_part9_v2',
    'HLT_HIMinimumBias_SinglePixelTrack_NpixGated_part9_v2'
)


# stream PhysicsHIMinimumBiasReducedFormat0

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIMinimumBiasReducedFormat0_datasetHIMinimumBiasReducedFormat0_selector
streamPhysicsHIMinimumBiasReducedFormat0_datasetHIMinimumBiasReducedFormat0_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIMinimumBiasReducedFormat0_datasetHIMinimumBiasReducedFormat0_selector.l1tResults = cms.InputTag('')
streamPhysicsHIMinimumBiasReducedFormat0_datasetHIMinimumBiasReducedFormat0_selector.throw      = cms.bool(False)
streamPhysicsHIMinimumBiasReducedFormat0_datasetHIMinimumBiasReducedFormat0_selector.triggerConditions = cms.vstring(
    'HLT_HIMinimumBiasRF_part0_v2',
    'HLT_HIMinimumBiasRF_part1_v2'
)


# stream PhysicsHIMinimumBiasReducedFormat1

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIMinimumBiasReducedFormat1_datasetHIMinimumBiasReducedFormat1_selector
streamPhysicsHIMinimumBiasReducedFormat1_datasetHIMinimumBiasReducedFormat1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIMinimumBiasReducedFormat1_datasetHIMinimumBiasReducedFormat1_selector.l1tResults = cms.InputTag('')
streamPhysicsHIMinimumBiasReducedFormat1_datasetHIMinimumBiasReducedFormat1_selector.throw      = cms.bool(False)
streamPhysicsHIMinimumBiasReducedFormat1_datasetHIMinimumBiasReducedFormat1_selector.triggerConditions = cms.vstring(
    'HLT_HIMinimumBiasRF_part2_v2',
    'HLT_HIMinimumBiasRF_part3_v2'
)


# stream PhysicsHIMinimumBiasReducedFormat10

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIMinimumBiasReducedFormat10_datasetHIMinimumBiasReducedFormat10_selector
streamPhysicsHIMinimumBiasReducedFormat10_datasetHIMinimumBiasReducedFormat10_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIMinimumBiasReducedFormat10_datasetHIMinimumBiasReducedFormat10_selector.l1tResults = cms.InputTag('')
streamPhysicsHIMinimumBiasReducedFormat10_datasetHIMinimumBiasReducedFormat10_selector.throw      = cms.bool(False)
streamPhysicsHIMinimumBiasReducedFormat10_datasetHIMinimumBiasReducedFormat10_selector.triggerConditions = cms.vstring(
    'HLT_HIMinimumBiasRF_part20_v2',
    'HLT_HIMinimumBiasRF_part21_v2'
)


# stream PhysicsHIMinimumBiasReducedFormat11

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIMinimumBiasReducedFormat11_datasetHIMinimumBiasReducedFormat11_selector
streamPhysicsHIMinimumBiasReducedFormat11_datasetHIMinimumBiasReducedFormat11_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIMinimumBiasReducedFormat11_datasetHIMinimumBiasReducedFormat11_selector.l1tResults = cms.InputTag('')
streamPhysicsHIMinimumBiasReducedFormat11_datasetHIMinimumBiasReducedFormat11_selector.throw      = cms.bool(False)
streamPhysicsHIMinimumBiasReducedFormat11_datasetHIMinimumBiasReducedFormat11_selector.triggerConditions = cms.vstring(
    'HLT_HIMinimumBiasRF_part22_v2',
    'HLT_HIMinimumBiasRF_part23_v2'
)


# stream PhysicsHIMinimumBiasReducedFormat2

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIMinimumBiasReducedFormat2_datasetHIMinimumBiasReducedFormat2_selector
streamPhysicsHIMinimumBiasReducedFormat2_datasetHIMinimumBiasReducedFormat2_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIMinimumBiasReducedFormat2_datasetHIMinimumBiasReducedFormat2_selector.l1tResults = cms.InputTag('')
streamPhysicsHIMinimumBiasReducedFormat2_datasetHIMinimumBiasReducedFormat2_selector.throw      = cms.bool(False)
streamPhysicsHIMinimumBiasReducedFormat2_datasetHIMinimumBiasReducedFormat2_selector.triggerConditions = cms.vstring(
    'HLT_HIMinimumBiasRF_part4_v2',
    'HLT_HIMinimumBiasRF_part5_v2'
)


# stream PhysicsHIMinimumBiasReducedFormat3

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIMinimumBiasReducedFormat3_datasetHIMinimumBiasReducedFormat3_selector
streamPhysicsHIMinimumBiasReducedFormat3_datasetHIMinimumBiasReducedFormat3_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIMinimumBiasReducedFormat3_datasetHIMinimumBiasReducedFormat3_selector.l1tResults = cms.InputTag('')
streamPhysicsHIMinimumBiasReducedFormat3_datasetHIMinimumBiasReducedFormat3_selector.throw      = cms.bool(False)
streamPhysicsHIMinimumBiasReducedFormat3_datasetHIMinimumBiasReducedFormat3_selector.triggerConditions = cms.vstring(
    'HLT_HIMinimumBiasRF_part6_v2',
    'HLT_HIMinimumBiasRF_part7_v2'
)


# stream PhysicsHIMinimumBiasReducedFormat4

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIMinimumBiasReducedFormat4_datasetHIMinimumBiasReducedFormat4_selector
streamPhysicsHIMinimumBiasReducedFormat4_datasetHIMinimumBiasReducedFormat4_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIMinimumBiasReducedFormat4_datasetHIMinimumBiasReducedFormat4_selector.l1tResults = cms.InputTag('')
streamPhysicsHIMinimumBiasReducedFormat4_datasetHIMinimumBiasReducedFormat4_selector.throw      = cms.bool(False)
streamPhysicsHIMinimumBiasReducedFormat4_datasetHIMinimumBiasReducedFormat4_selector.triggerConditions = cms.vstring(
    'HLT_HIMinimumBiasRF_part8_v2',
    'HLT_HIMinimumBiasRF_part9_v2'
)


# stream PhysicsHIMinimumBiasReducedFormat5

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIMinimumBiasReducedFormat5_datasetHIMinimumBiasReducedFormat5_selector
streamPhysicsHIMinimumBiasReducedFormat5_datasetHIMinimumBiasReducedFormat5_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIMinimumBiasReducedFormat5_datasetHIMinimumBiasReducedFormat5_selector.l1tResults = cms.InputTag('')
streamPhysicsHIMinimumBiasReducedFormat5_datasetHIMinimumBiasReducedFormat5_selector.throw      = cms.bool(False)
streamPhysicsHIMinimumBiasReducedFormat5_datasetHIMinimumBiasReducedFormat5_selector.triggerConditions = cms.vstring(
    'HLT_HIMinimumBiasRF_part10_v2',
    'HLT_HIMinimumBiasRF_part11_v2'
)


# stream PhysicsHIMinimumBiasReducedFormat6

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIMinimumBiasReducedFormat6_datasetHIMinimumBiasReducedFormat6_selector
streamPhysicsHIMinimumBiasReducedFormat6_datasetHIMinimumBiasReducedFormat6_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIMinimumBiasReducedFormat6_datasetHIMinimumBiasReducedFormat6_selector.l1tResults = cms.InputTag('')
streamPhysicsHIMinimumBiasReducedFormat6_datasetHIMinimumBiasReducedFormat6_selector.throw      = cms.bool(False)
streamPhysicsHIMinimumBiasReducedFormat6_datasetHIMinimumBiasReducedFormat6_selector.triggerConditions = cms.vstring(
    'HLT_HIMinimumBiasRF_part12_v2',
    'HLT_HIMinimumBiasRF_part13_v2'
)


# stream PhysicsHIMinimumBiasReducedFormat7

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIMinimumBiasReducedFormat7_datasetHIMinimumBiasReducedFormat7_selector
streamPhysicsHIMinimumBiasReducedFormat7_datasetHIMinimumBiasReducedFormat7_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIMinimumBiasReducedFormat7_datasetHIMinimumBiasReducedFormat7_selector.l1tResults = cms.InputTag('')
streamPhysicsHIMinimumBiasReducedFormat7_datasetHIMinimumBiasReducedFormat7_selector.throw      = cms.bool(False)
streamPhysicsHIMinimumBiasReducedFormat7_datasetHIMinimumBiasReducedFormat7_selector.triggerConditions = cms.vstring(
    'HLT_HIMinimumBiasRF_part14_v2',
    'HLT_HIMinimumBiasRF_part15_v2'
)


# stream PhysicsHIMinimumBiasReducedFormat8

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIMinimumBiasReducedFormat8_datasetHIMinimumBiasReducedFormat8_selector
streamPhysicsHIMinimumBiasReducedFormat8_datasetHIMinimumBiasReducedFormat8_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIMinimumBiasReducedFormat8_datasetHIMinimumBiasReducedFormat8_selector.l1tResults = cms.InputTag('')
streamPhysicsHIMinimumBiasReducedFormat8_datasetHIMinimumBiasReducedFormat8_selector.throw      = cms.bool(False)
streamPhysicsHIMinimumBiasReducedFormat8_datasetHIMinimumBiasReducedFormat8_selector.triggerConditions = cms.vstring(
    'HLT_HIMinimumBiasRF_part16_v2',
    'HLT_HIMinimumBiasRF_part17_v2'
)


# stream PhysicsHIMinimumBiasReducedFormat9

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIMinimumBiasReducedFormat9_datasetHIMinimumBiasReducedFormat9_selector
streamPhysicsHIMinimumBiasReducedFormat9_datasetHIMinimumBiasReducedFormat9_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIMinimumBiasReducedFormat9_datasetHIMinimumBiasReducedFormat9_selector.l1tResults = cms.InputTag('')
streamPhysicsHIMinimumBiasReducedFormat9_datasetHIMinimumBiasReducedFormat9_selector.throw      = cms.bool(False)
streamPhysicsHIMinimumBiasReducedFormat9_datasetHIMinimumBiasReducedFormat9_selector.triggerConditions = cms.vstring(
    'HLT_HIMinimumBiasRF_part18_v2',
    'HLT_HIMinimumBiasRF_part19_v2'
)


# stream PhysicsHISingleMuon

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHISingleMuon_datasetHISingleMuon_selector
streamPhysicsHISingleMuon_datasetHISingleMuon_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHISingleMuon_datasetHISingleMuon_selector.l1tResults = cms.InputTag('')
streamPhysicsHISingleMuon_datasetHISingleMuon_selector.throw      = cms.bool(False)
streamPhysicsHISingleMuon_datasetHISingleMuon_selector.triggerConditions = cms.vstring(
    'HLT_HIL1MuOpen_Centrality_70_100_v2',
    'HLT_HIL1MuOpen_Centrality_80_100_v2',
    'HLT_HIL2Mu3_NHitQ15_v2',
    'HLT_HIL2Mu5_NHitQ15_v2',
    'HLT_HIL2Mu7_NHitQ15_v2',
    'HLT_HIL3Mu12_v2',
    'HLT_HIL3Mu15_v2',
    'HLT_HIL3Mu20_v2',
    'HLT_HIL3Mu3Eta2p5_PuAK4CaloJet100Eta2p1_FilterDr_v2',
    'HLT_HIL3Mu3Eta2p5_PuAK4CaloJet100Eta2p1_v2',
    'HLT_HIL3Mu3Eta2p5_PuAK4CaloJet40Eta2p1_FilterDr_v2',
    'HLT_HIL3Mu3Eta2p5_PuAK4CaloJet40Eta2p1_v2',
    'HLT_HIL3Mu3Eta2p5_PuAK4CaloJet60Eta2p1_FilterDr_v2',
    'HLT_HIL3Mu3Eta2p5_PuAK4CaloJet60Eta2p1_v2',
    'HLT_HIL3Mu3Eta2p5_PuAK4CaloJet80Eta2p1_FilterDr_v2',
    'HLT_HIL3Mu3Eta2p5_PuAK4CaloJet80Eta2p1_v2',
    'HLT_HIL3Mu3_NHitQ10_v2',
    'HLT_HIL3Mu5Eta2p5_PuAK4CaloJet100Eta2p1_FilterDr_v2',
    'HLT_HIL3Mu5Eta2p5_PuAK4CaloJet100Eta2p1_v2',
    'HLT_HIL3Mu5Eta2p5_PuAK4CaloJet40Eta2p1_FilterDr_v2',
    'HLT_HIL3Mu5Eta2p5_PuAK4CaloJet40Eta2p1_v2',
    'HLT_HIL3Mu5Eta2p5_PuAK4CaloJet60Eta2p1_FilterDr_v2',
    'HLT_HIL3Mu5Eta2p5_PuAK4CaloJet60Eta2p1_v2',
    'HLT_HIL3Mu5Eta2p5_PuAK4CaloJet80Eta2p1_FilterDr_v2',
    'HLT_HIL3Mu5Eta2p5_PuAK4CaloJet80Eta2p1_v2',
    'HLT_HIL3Mu5_NHitQ10_v2',
    'HLT_HIL3Mu7_NHitQ10_v2'
)


# stream PhysicsHITrackerNZS

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHITrackerNZS_datasetHITrackerNZS_selector
streamPhysicsHITrackerNZS_datasetHITrackerNZS_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHITrackerNZS_datasetHITrackerNZS_selector.l1tResults = cms.InputTag('')
streamPhysicsHITrackerNZS_datasetHITrackerNZS_selector.throw      = cms.bool(False)
streamPhysicsHITrackerNZS_datasetHITrackerNZS_selector.triggerConditions = cms.vstring('HLT_HIPhysicsForZS_v2')

