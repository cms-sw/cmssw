# /dev/CMSSW_13_2_0/HIon

import FWCore.ParameterSet.Config as cms


# stream PhysicsHICommissioning

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHICommissioning_datasetHIEmptyBX_selector
streamPhysicsHICommissioning_datasetHIEmptyBX_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHICommissioning_datasetHIEmptyBX_selector.l1tResults = cms.InputTag('')
streamPhysicsHICommissioning_datasetHIEmptyBX_selector.throw      = cms.bool(False)
streamPhysicsHICommissioning_datasetHIEmptyBX_selector.triggerConditions = cms.vstring(
    'HLT_HIL1NotBptxOR_v7',
    'HLT_HIL1UnpairedBunchBptxMinus_v7',
    'HLT_HIL1UnpairedBunchBptxPlus_v7'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHICommissioning_datasetHIHLTPhysics_selector
streamPhysicsHICommissioning_datasetHIHLTPhysics_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHICommissioning_datasetHIHLTPhysics_selector.l1tResults = cms.InputTag('')
streamPhysicsHICommissioning_datasetHIHLTPhysics_selector.throw      = cms.bool(False)
streamPhysicsHICommissioning_datasetHIHLTPhysics_selector.triggerConditions = cms.vstring('HLT_HIPhysics_v7')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHICommissioning_datasetHIHcalNZS_selector
streamPhysicsHICommissioning_datasetHIHcalNZS_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHICommissioning_datasetHIHcalNZS_selector.l1tResults = cms.InputTag('')
streamPhysicsHICommissioning_datasetHIHcalNZS_selector.throw      = cms.bool(False)
streamPhysicsHICommissioning_datasetHIHcalNZS_selector.triggerConditions = cms.vstring(
    'HLT_HIHcalNZS_v7',
    'HLT_HIHcalPhiSym_v7'
)


# stream PhysicsHIDoubleMuon

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIDoubleMuon_datasetHIDoubleMuon_selector
streamPhysicsHIDoubleMuon_datasetHIDoubleMuon_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIDoubleMuon_datasetHIDoubleMuon_selector.l1tResults = cms.InputTag('')
streamPhysicsHIDoubleMuon_datasetHIDoubleMuon_selector.throw      = cms.bool(False)
streamPhysicsHIDoubleMuon_datasetHIDoubleMuon_selector.triggerConditions = cms.vstring(
    'HLT_HIL1DoubleMu0_v4',
    'HLT_HIL1DoubleMu10_v4',
    'HLT_HIL1DoubleMuOpen_OS_er1p6_v4',
    'HLT_HIL1DoubleMuOpen_er1p6_v4',
    'HLT_HIL1DoubleMuOpen_v4',
    'HLT_HIL2DoubleMuOpen_v6',
    'HLT_HIL2_L1DoubleMu10_v6',
    'HLT_HIL3DoubleMuOpen_v6',
    'HLT_HIL3Mu0_L2Mu0_v6',
    'HLT_HIL3Mu2p5NHitQ10_L2Mu2_M7toinf_v6',
    'HLT_HIL3Mu2p5NHitQ10_L2Mu2_v6',
    'HLT_HIL3Mu2p5_L1DoubleMu0_v6',
    'HLT_HIL3Mu3NHitQ10_L1DoubleMuOpen_v6',
    'HLT_HIL3Mu3_L1DoubleMuOpen_OS_v6',
    'HLT_HIL3Mu3_L1TripleMuOpen_v6',
    'HLT_HIL3_L1DoubleMu10_v6'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIDoubleMuon_datasetHIDoubleMuonPsiPeri_selector
streamPhysicsHIDoubleMuon_datasetHIDoubleMuonPsiPeri_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIDoubleMuon_datasetHIDoubleMuonPsiPeri_selector.l1tResults = cms.InputTag('')
streamPhysicsHIDoubleMuon_datasetHIDoubleMuonPsiPeri_selector.throw      = cms.bool(False)
streamPhysicsHIDoubleMuon_datasetHIDoubleMuonPsiPeri_selector.triggerConditions = cms.vstring(
    'HLT_HIL1DoubleMuOpen_Centrality_30_100_v4',
    'HLT_HIL1DoubleMuOpen_Centrality_40_100_v4',
    'HLT_HIL1DoubleMuOpen_Centrality_50_100_v4',
    'HLT_HIL3Mu0NHitQ10_L2Mu0_MAXdR3p5_M1to5_v6'
)


# stream PhysicsHIForward

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIForward_datasetHIForward_selector
streamPhysicsHIForward_datasetHIForward_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIForward_datasetHIForward_selector.l1tResults = cms.InputTag('')
streamPhysicsHIForward_datasetHIForward_selector.throw      = cms.bool(False)
streamPhysicsHIForward_datasetHIForward_selector.triggerConditions = cms.vstring(
    'HLT_HIUPC_DoubleEG2_BptxAND_SinglePixelTrack_MaxPixelTrack_v7',
    'HLT_HIUPC_DoubleEG2_NotMBHF2AND_SinglePixelTrack_MaxPixelTrack_v7',
    'HLT_HIUPC_DoubleEG2_NotMBHF2AND_v7',
    'HLT_HIUPC_DoubleEG5_BptxAND_SinglePixelTrack_MaxPixelTrack_v7',
    'HLT_HIUPC_DoubleEG5_NotMBHF2AND_SinglePixelTrack_MaxPixelTrack_v7',
    'HLT_HIUPC_DoubleEG5_NotMBHF2AND_v7',
    'HLT_HIUPC_DoubleMuCosmic_BptxAND_MaxPixelTrack_v5',
    'HLT_HIUPC_DoubleMuCosmic_NotMBHF2AND_MaxPixelTrack_v5',
    'HLT_HIUPC_DoubleMuCosmic_NotMBHF2AND_v5',
    'HLT_HIUPC_DoubleMuOpen_BptxAND_MaxPixelTrack_v7',
    'HLT_HIUPC_DoubleMuOpen_NotMBHF2AND_MaxPixelTrack_v5',
    'HLT_HIUPC_DoubleMuOpen_NotMBHF2AND_v5',
    'HLT_HIUPC_MBHF1AND_PixelTrackMultiplicity20400_v5',
    'HLT_HIUPC_MBHF1AND_PixelTrackMultiplicity30400_v5',
    'HLT_HIUPC_MBHF1AND_PixelTrackMultiplicity40400_v5',
    'HLT_HIUPC_MBHF2AND_PixelTrackMultiplicity20400_v5',
    'HLT_HIUPC_MBHF2AND_PixelTrackMultiplicity30400_v5',
    'HLT_HIUPC_MBHF2AND_PixelTrackMultiplicity40400_v5',
    'HLT_HIUPC_SingleEG2_NotMBHF2AND_ZDC1nOR_SinglePixelTrack_MaxPixelTrack_v5',
    'HLT_HIUPC_SingleEG3_BptxAND_SinglePixelTrack_MaxPixelTrack_v7',
    'HLT_HIUPC_SingleEG3_NotMBHF2AND_SinglePixelTrack_MaxPixelTrack_v7',
    'HLT_HIUPC_SingleEG3_NotMBHF2AND_v7',
    'HLT_HIUPC_SingleEG3_NotMBHF2OR_SinglePixelTrack_MaxPixelTrack_v7',
    'HLT_HIUPC_SingleEG3_NotMBHF2OR_v7',
    'HLT_HIUPC_SingleEG5_BptxAND_SinglePixelTrack_MaxPixelTrack_v7',
    'HLT_HIUPC_SingleEG5_NotMBHF2AND_SinglePixelTrack_MaxPixelTrack_v7',
    'HLT_HIUPC_SingleEG5_NotMBHF2AND_v7',
    'HLT_HIUPC_SingleMuCosmic_BptxAND_MaxPixelTrack_v5',
    'HLT_HIUPC_SingleMuCosmic_NotMBHF2AND_MaxPixelTrack_v5',
    'HLT_HIUPC_SingleMuCosmic_NotMBHF2AND_v5',
    'HLT_HIUPC_SingleMuCosmic_NotMBHF2OR_MaxPixelTrack_v5',
    'HLT_HIUPC_SingleMuCosmic_NotMBHF2OR_v5',
    'HLT_HIUPC_SingleMuOpen_BptxAND_MaxPixelTrack_v7',
    'HLT_HIUPC_SingleMuOpen_NotMBHF2AND_MaxPixelTrack_v7',
    'HLT_HIUPC_SingleMuOpen_NotMBHF2AND_v7',
    'HLT_HIUPC_SingleMuOpen_NotMBHF2OR_MaxPixelTrack_v7',
    'HLT_HIUPC_SingleMuOpen_NotMBHF2OR_v7',
    'HLT_HIUPC_SingleMuOpen_OR_SingleMuCosmic_EMTF_BptxAND_MaxPixelTrack_v5',
    'HLT_HIUPC_SingleMuOpen_OR_SingleMuCosmic_EMTF_NotMBHF2AND_MaxPixelTrack_v5',
    'HLT_HIUPC_SingleMuOpen_OR_SingleMuCosmic_EMTF_NotMBHF2AND_v5',
    'HLT_HIUPC_SingleMuOpen_OR_SingleMuCosmic_EMTF_NotMBHF2OR_MaxPixelTrack_v5',
    'HLT_HIUPC_SingleMuOpen_OR_SingleMuCosmic_EMTF_NotMBHF2OR_v5',
    'HLT_HIUPC_ZDC1nOR_MinPixelCluster400_MaxPixelCluster10000_v5',
    'HLT_HIUPC_ZDC1nOR_SinglePixelTrackLowPt_MaxPixelCluster400_v5',
    'HLT_HIUPC_ZDC1nOR_SinglePixelTrack_MaxPixelTrack_v5',
    'HLT_HIUPC_ZDC1nXOR_MBHF1AND_PixelTrackMultiplicity20_v5',
    'HLT_HIUPC_ZDC1nXOR_MBHF1AND_PixelTrackMultiplicity30_v5',
    'HLT_HIUPC_ZDC1nXOR_MBHF1AND_PixelTrackMultiplicity40_v5',
    'HLT_HIUPC_ZDC1nXOR_MBHF2AND_PixelTrackMultiplicity20_v5',
    'HLT_HIUPC_ZDC1nXOR_MBHF2AND_PixelTrackMultiplicity30_v5',
    'HLT_HIUPC_ZDC1nXOR_MBHF2AND_PixelTrackMultiplicity40_v5',
    'HLT_HIUPC_ZeroBias_MinPixelCluster400_MaxPixelCluster10000_v5',
    'HLT_HIUPC_ZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v5',
    'HLT_HIUPC_ZeroBias_SinglePixelTrack_MaxPixelTrack_v5',
    'HLT_HIZeroBias_v7'
)


# stream PhysicsHIHardProbes

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIHardProbes_datasetHIHardProbes_selector
streamPhysicsHIHardProbes_datasetHIHardProbes_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIHardProbes_datasetHIHardProbes_selector.l1tResults = cms.InputTag('')
streamPhysicsHIHardProbes_datasetHIHardProbes_selector.throw      = cms.bool(False)
streamPhysicsHIHardProbes_datasetHIHardProbes_selector.triggerConditions = cms.vstring(
    'HLT_HICsAK4PFJet100Eta1p5_v7',
    'HLT_HICsAK4PFJet120Eta1p5_v7',
    'HLT_HICsAK4PFJet80Eta1p5_v7',
    'HLT_HIDoubleEle10GsfMass50_v7',
    'HLT_HIDoubleEle10Gsf_v7',
    'HLT_HIDoubleEle15GsfMass50_v7',
    'HLT_HIDoubleEle15Gsf_v7',
    'HLT_HIEle10Gsf_v7',
    'HLT_HIEle15Ele10GsfMass50_v7',
    'HLT_HIEle15Ele10Gsf_v7',
    'HLT_HIEle15Gsf_v7',
    'HLT_HIEle20Gsf_v7',
    'HLT_HIEle30Gsf_v7',
    'HLT_HIEle40Gsf_v7',
    'HLT_HIEle50Gsf_v7',
    'HLT_HIGEDPhoton30_EB_HECut_v7',
    'HLT_HIGEDPhoton30_EB_v7',
    'HLT_HIGEDPhoton30_HECut_v7',
    'HLT_HIGEDPhoton30_v7',
    'HLT_HIGEDPhoton40_EB_HECut_v7',
    'HLT_HIGEDPhoton40_EB_v7',
    'HLT_HIGEDPhoton40_HECut_v7',
    'HLT_HIGEDPhoton40_v7',
    'HLT_HIGEDPhoton50_EB_HECut_v7',
    'HLT_HIGEDPhoton50_EB_v7',
    'HLT_HIGEDPhoton50_HECut_v7',
    'HLT_HIGEDPhoton50_v7',
    'HLT_HIGEDPhoton60_EB_HECut_v7',
    'HLT_HIGEDPhoton60_EB_v7',
    'HLT_HIGEDPhoton60_HECut_v7',
    'HLT_HIGEDPhoton60_v7',
    'HLT_HIL1Mu3Eta2p5_Ele10Gsf_v7',
    'HLT_HIL1Mu3Eta2p5_Ele15Gsf_v7',
    'HLT_HIL1Mu3Eta2p5_Ele20Gsf_v7',
    'HLT_HIL1Mu5Eta2p5_Ele10Gsf_v7',
    'HLT_HIL1Mu5Eta2p5_Ele15Gsf_v7',
    'HLT_HIL1Mu5Eta2p5_Ele20Gsf_v7',
    'HLT_HIL1Mu7Eta2p5_Ele10Gsf_v7',
    'HLT_HIL1Mu7Eta2p5_Ele15Gsf_v7',
    'HLT_HIL1Mu7Eta2p5_Ele20Gsf_v7',
    'HLT_HIL3Mu3_EG10HECut_v7',
    'HLT_HIL3Mu3_EG15HECut_v7',
    'HLT_HIL3Mu3_EG20HECut_v7',
    'HLT_HIL3Mu3_EG30HECut_v7',
    'HLT_HIL3Mu5_EG10HECut_v7',
    'HLT_HIL3Mu5_EG15HECut_v7',
    'HLT_HIL3Mu5_EG20HECut_v7',
    'HLT_HIL3Mu5_EG30HECut_v7',
    'HLT_HIL3Mu7_EG10HECut_v7',
    'HLT_HIL3Mu7_EG15HECut_v7',
    'HLT_HIL3Mu7_EG20HECut_v7',
    'HLT_HIL3Mu7_EG30HECut_v7',
    'HLT_HIPuAK4CaloJet100Eta5p1_v7',
    'HLT_HIPuAK4CaloJet120Eta5p1_v7',
    'HLT_HIPuAK4CaloJet80Eta5p1_v7'
)


# stream PhysicsHIHardProbesLower

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIHardProbesLower_datasetHIHardProbesLower_selector
streamPhysicsHIHardProbesLower_datasetHIHardProbesLower_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIHardProbesLower_datasetHIHardProbesLower_selector.l1tResults = cms.InputTag('')
streamPhysicsHIHardProbesLower_datasetHIHardProbesLower_selector.throw      = cms.bool(False)
streamPhysicsHIHardProbesLower_datasetHIHardProbesLower_selector.triggerConditions = cms.vstring(
    'HLT_HICsAK4PFJet60Eta1p5_v7',
    'HLT_HIGEDPhoton10_EB_HECut_v7',
    'HLT_HIGEDPhoton10_EB_v7',
    'HLT_HIGEDPhoton10_HECut_v7',
    'HLT_HIGEDPhoton10_v7',
    'HLT_HIGEDPhoton20_EB_HECut_v7',
    'HLT_HIGEDPhoton20_EB_v7',
    'HLT_HIGEDPhoton20_HECut_v7',
    'HLT_HIGEDPhoton20_v7',
    'HLT_HIPuAK4CaloJet40Eta5p1_v7',
    'HLT_HIPuAK4CaloJet60Eta5p1_v7'
)


# stream PhysicsHIHardProbesPeripheral

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIHardProbesPeripheral_datasetHIHardProbesPeripheral_selector
streamPhysicsHIHardProbesPeripheral_datasetHIHardProbesPeripheral_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIHardProbesPeripheral_datasetHIHardProbesPeripheral_selector.l1tResults = cms.InputTag('')
streamPhysicsHIHardProbesPeripheral_datasetHIHardProbesPeripheral_selector.throw      = cms.bool(False)
streamPhysicsHIHardProbesPeripheral_datasetHIHardProbesPeripheral_selector.triggerConditions = cms.vstring(
    'HLT_HICsAK4PFJet100Eta1p5_Centrality_30_100_v7',
    'HLT_HICsAK4PFJet60Eta1p5_Centrality_30_100_v7',
    'HLT_HICsAK4PFJet80Eta1p5_Centrality_30_100_v7',
    'HLT_HIGEDPhoton10_Cent30_100_v7',
    'HLT_HIGEDPhoton20_Cent30_100_v7',
    'HLT_HIGEDPhoton30_Cent30_100_v7',
    'HLT_HIGEDPhoton40_Cent30_100_v7',
    'HLT_HIPuAK4CaloJet100Eta5p1_Centrality_30_100_v7',
    'HLT_HIPuAK4CaloJet40Eta5p1_Centrality_30_100_v7',
    'HLT_HIPuAK4CaloJet60Eta5p1_Centrality_30_100_v7',
    'HLT_HIPuAK4CaloJet80Eta5p1_Centrality_30_100_v7'
)


# stream PhysicsHIHeavyFlavor

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIHeavyFlavor_datasetHIHeavyFlavor_selector
streamPhysicsHIHeavyFlavor_datasetHIHeavyFlavor_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIHeavyFlavor_datasetHIHeavyFlavor_selector.l1tResults = cms.InputTag('')
streamPhysicsHIHeavyFlavor_datasetHIHeavyFlavor_selector.throw      = cms.bool(False)
streamPhysicsHIHeavyFlavor_datasetHIHeavyFlavor_selector.triggerConditions = cms.vstring(
    'HLT_HIDmesonPPTrackingGlobal_Dpt20_NoIter10_v7',
    'HLT_HIDmesonPPTrackingGlobal_Dpt20_v7',
    'HLT_HIDmesonPPTrackingGlobal_Dpt30_NoIter10_v7',
    'HLT_HIDmesonPPTrackingGlobal_Dpt30_v7',
    'HLT_HIDmesonPPTrackingGlobal_Dpt40_NoIter10_v7',
    'HLT_HIDmesonPPTrackingGlobal_Dpt40_v7',
    'HLT_HIDmesonPPTrackingGlobal_Dpt50_NoIter10_v7',
    'HLT_HIDmesonPPTrackingGlobal_Dpt50_v7',
    'HLT_HIDmesonPPTrackingGlobal_Dpt60_NoIter10_v7',
    'HLT_HIDmesonPPTrackingGlobal_Dpt60_v7',
    'HLT_HIDsPPTrackingGlobal_Dpt20_NoIter10_v7',
    'HLT_HIDsPPTrackingGlobal_Dpt20_v7',
    'HLT_HIDsPPTrackingGlobal_Dpt30_NoIter10_v7',
    'HLT_HIDsPPTrackingGlobal_Dpt30_v7',
    'HLT_HIDsPPTrackingGlobal_Dpt40_NoIter10_v7',
    'HLT_HIDsPPTrackingGlobal_Dpt40_v7',
    'HLT_HIDsPPTrackingGlobal_Dpt50_NoIter10_v7',
    'HLT_HIDsPPTrackingGlobal_Dpt50_v7',
    'HLT_HIDsPPTrackingGlobal_Dpt60_NoIter10_v7',
    'HLT_HIDsPPTrackingGlobal_Dpt60_v7',
    'HLT_HIFullTracks2018_HighPt18_NoIter10_v7',
    'HLT_HIFullTracks2018_HighPt18_v7',
    'HLT_HIFullTracks2018_HighPt24_NoIter10_v7',
    'HLT_HIFullTracks2018_HighPt24_v7',
    'HLT_HIFullTracks2018_HighPt34_NoIter10_v7',
    'HLT_HIFullTracks2018_HighPt34_v7',
    'HLT_HIFullTracks2018_HighPt45_NoIter10_v7',
    'HLT_HIFullTracks2018_HighPt45_v7',
    'HLT_HIFullTracks2018_HighPt56_NoIter10_v7',
    'HLT_HIFullTracks2018_HighPt56_v7',
    'HLT_HIFullTracks2018_HighPt60_NoIter10_v7',
    'HLT_HIFullTracks2018_HighPt60_v7',
    'HLT_HILcPPTrackingGlobal_Dpt20_NoIter10_v7',
    'HLT_HILcPPTrackingGlobal_Dpt20_v7',
    'HLT_HILcPPTrackingGlobal_Dpt30_NoIter10_v7',
    'HLT_HILcPPTrackingGlobal_Dpt30_v7',
    'HLT_HILcPPTrackingGlobal_Dpt40_NoIter10_v7',
    'HLT_HILcPPTrackingGlobal_Dpt40_v7',
    'HLT_HILcPPTrackingGlobal_Dpt50_NoIter10_v7',
    'HLT_HILcPPTrackingGlobal_Dpt50_v7',
    'HLT_HILcPPTrackingGlobal_Dpt60_NoIter10_v7',
    'HLT_HILcPPTrackingGlobal_Dpt60_v7'
)


# stream PhysicsHISingleMuon

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHISingleMuon_datasetHISingleMuon_selector
streamPhysicsHISingleMuon_datasetHISingleMuon_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHISingleMuon_datasetHISingleMuon_selector.l1tResults = cms.InputTag('')
streamPhysicsHISingleMuon_datasetHISingleMuon_selector.throw      = cms.bool(False)
streamPhysicsHISingleMuon_datasetHISingleMuon_selector.triggerConditions = cms.vstring(
    'HLT_HIL2Mu3_NHitQ15_v6',
    'HLT_HIL2Mu5_NHitQ15_v6',
    'HLT_HIL2Mu7_NHitQ15_v6',
    'HLT_HIL3Mu12_v6',
    'HLT_HIL3Mu15_v6',
    'HLT_HIL3Mu20_v6',
    'HLT_HIL3Mu3Eta2p5_PuAK4CaloJet100Eta2p1_FilterDr_v7',
    'HLT_HIL3Mu3Eta2p5_PuAK4CaloJet100Eta2p1_v7',
    'HLT_HIL3Mu3Eta2p5_PuAK4CaloJet40Eta2p1_FilterDr_v7',
    'HLT_HIL3Mu3Eta2p5_PuAK4CaloJet40Eta2p1_v7',
    'HLT_HIL3Mu3Eta2p5_PuAK4CaloJet60Eta2p1_FilterDr_v7',
    'HLT_HIL3Mu3Eta2p5_PuAK4CaloJet60Eta2p1_v7',
    'HLT_HIL3Mu3Eta2p5_PuAK4CaloJet80Eta2p1_FilterDr_v7',
    'HLT_HIL3Mu3Eta2p5_PuAK4CaloJet80Eta2p1_v7',
    'HLT_HIL3Mu3_NHitQ10_v6',
    'HLT_HIL3Mu5Eta2p5_PuAK4CaloJet100Eta2p1_FilterDr_v7',
    'HLT_HIL3Mu5Eta2p5_PuAK4CaloJet100Eta2p1_v7',
    'HLT_HIL3Mu5Eta2p5_PuAK4CaloJet40Eta2p1_FilterDr_v7',
    'HLT_HIL3Mu5Eta2p5_PuAK4CaloJet40Eta2p1_v7',
    'HLT_HIL3Mu5Eta2p5_PuAK4CaloJet60Eta2p1_FilterDr_v7',
    'HLT_HIL3Mu5Eta2p5_PuAK4CaloJet60Eta2p1_v7',
    'HLT_HIL3Mu5Eta2p5_PuAK4CaloJet80Eta2p1_FilterDr_v7',
    'HLT_HIL3Mu5Eta2p5_PuAK4CaloJet80Eta2p1_v7',
    'HLT_HIL3Mu5_NHitQ10_v6',
    'HLT_HIL3Mu7_NHitQ10_v6'
)


# stream PhysicsHITestRaw

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHITestRaw_datasetHITestRaw_selector
streamPhysicsHITestRaw_datasetHITestRaw_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHITestRaw_datasetHITestRaw_selector.l1tResults = cms.InputTag('')
streamPhysicsHITestRaw_datasetHITestRaw_selector.throw      = cms.bool(False)
streamPhysicsHITestRaw_datasetHITestRaw_selector.triggerConditions = cms.vstring(
    'HLT_HIMinimumBias_v5',
    'HLT_HIRandom_v5',
    'HLT_HIUPC_DoubleEG2_BptxAND_SinglePixelTrack_MaxPixelTrack_v7',
    'HLT_HIUPC_DoubleEG2_NotMBHF2AND_SinglePixelTrack_MaxPixelTrack_v7',
    'HLT_HIUPC_DoubleEG2_NotMBHF2AND_v7',
    'HLT_HIUPC_DoubleEG5_BptxAND_SinglePixelTrack_MaxPixelTrack_v7',
    'HLT_HIUPC_DoubleEG5_NotMBHF2AND_SinglePixelTrack_MaxPixelTrack_v7',
    'HLT_HIUPC_DoubleEG5_NotMBHF2AND_v7',
    'HLT_HIUPC_DoubleMuCosmic_BptxAND_MaxPixelTrack_v5',
    'HLT_HIUPC_DoubleMuCosmic_NotMBHF2AND_MaxPixelTrack_v5',
    'HLT_HIUPC_DoubleMuCosmic_NotMBHF2AND_v5',
    'HLT_HIUPC_DoubleMuOpen_BptxAND_MaxPixelTrack_v7',
    'HLT_HIUPC_DoubleMuOpen_NotMBHF2AND_MaxPixelTrack_v5',
    'HLT_HIUPC_DoubleMuOpen_NotMBHF2AND_v5',
    'HLT_HIUPC_MBHF1AND_PixelTrackMultiplicity20400_v5',
    'HLT_HIUPC_MBHF1AND_PixelTrackMultiplicity30400_v5',
    'HLT_HIUPC_MBHF1AND_PixelTrackMultiplicity40400_v5',
    'HLT_HIUPC_MBHF2AND_PixelTrackMultiplicity20400_v5',
    'HLT_HIUPC_MBHF2AND_PixelTrackMultiplicity30400_v5',
    'HLT_HIUPC_MBHF2AND_PixelTrackMultiplicity40400_v5',
    'HLT_HIUPC_SingleEG2_NotMBHF2AND_ZDC1nOR_SinglePixelTrack_MaxPixelTrack_v5',
    'HLT_HIUPC_SingleEG3_BptxAND_SinglePixelTrack_MaxPixelTrack_v7',
    'HLT_HIUPC_SingleEG3_NotMBHF2AND_SinglePixelTrack_MaxPixelTrack_v7',
    'HLT_HIUPC_SingleEG3_NotMBHF2AND_v7',
    'HLT_HIUPC_SingleEG3_NotMBHF2OR_SinglePixelTrack_MaxPixelTrack_v7',
    'HLT_HIUPC_SingleEG3_NotMBHF2OR_v7',
    'HLT_HIUPC_SingleEG5_BptxAND_SinglePixelTrack_MaxPixelTrack_v7',
    'HLT_HIUPC_SingleEG5_NotMBHF2AND_SinglePixelTrack_MaxPixelTrack_v7',
    'HLT_HIUPC_SingleEG5_NotMBHF2AND_v7',
    'HLT_HIUPC_SingleMuCosmic_BptxAND_MaxPixelTrack_v5',
    'HLT_HIUPC_SingleMuCosmic_NotMBHF2AND_MaxPixelTrack_v5',
    'HLT_HIUPC_SingleMuCosmic_NotMBHF2AND_v5',
    'HLT_HIUPC_SingleMuCosmic_NotMBHF2OR_MaxPixelTrack_v5',
    'HLT_HIUPC_SingleMuCosmic_NotMBHF2OR_v5',
    'HLT_HIUPC_SingleMuOpen_BptxAND_MaxPixelTrack_v7',
    'HLT_HIUPC_SingleMuOpen_NotMBHF2AND_MaxPixelTrack_v7',
    'HLT_HIUPC_SingleMuOpen_NotMBHF2AND_v7',
    'HLT_HIUPC_SingleMuOpen_NotMBHF2OR_MaxPixelTrack_v7',
    'HLT_HIUPC_SingleMuOpen_NotMBHF2OR_v7',
    'HLT_HIUPC_SingleMuOpen_OR_SingleMuCosmic_EMTF_BptxAND_MaxPixelTrack_v5',
    'HLT_HIUPC_SingleMuOpen_OR_SingleMuCosmic_EMTF_NotMBHF2AND_MaxPixelTrack_v5',
    'HLT_HIUPC_SingleMuOpen_OR_SingleMuCosmic_EMTF_NotMBHF2AND_v5',
    'HLT_HIUPC_SingleMuOpen_OR_SingleMuCosmic_EMTF_NotMBHF2OR_MaxPixelTrack_v5',
    'HLT_HIUPC_SingleMuOpen_OR_SingleMuCosmic_EMTF_NotMBHF2OR_v5',
    'HLT_HIUPC_ZeroBias_MinPixelCluster400_MaxPixelCluster10000_v5',
    'HLT_HIUPC_ZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v5',
    'HLT_HIUPC_ZeroBias_SinglePixelTrack_MaxPixelTrack_v5',
    'HLT_HIZeroBias_v7'
)


# stream PhysicsHITestRawPrime

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHITestRawPrime_datasetHITestRawPrime_selector
streamPhysicsHITestRawPrime_datasetHITestRawPrime_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHITestRawPrime_datasetHITestRawPrime_selector.l1tResults = cms.InputTag('')
streamPhysicsHITestRawPrime_datasetHITestRawPrime_selector.throw      = cms.bool(False)
streamPhysicsHITestRawPrime_datasetHITestRawPrime_selector.triggerConditions = cms.vstring(
    'HLT_HIMinimumBias_v5',
    'HLT_HIRandom_v5',
    'HLT_HIUPC_DoubleEG2_BptxAND_SinglePixelTrack_MaxPixelTrack_v7',
    'HLT_HIUPC_DoubleEG2_NotMBHF2AND_SinglePixelTrack_MaxPixelTrack_v7',
    'HLT_HIUPC_DoubleEG2_NotMBHF2AND_v7',
    'HLT_HIUPC_DoubleEG5_BptxAND_SinglePixelTrack_MaxPixelTrack_v7',
    'HLT_HIUPC_DoubleEG5_NotMBHF2AND_SinglePixelTrack_MaxPixelTrack_v7',
    'HLT_HIUPC_DoubleEG5_NotMBHF2AND_v7',
    'HLT_HIUPC_DoubleMuCosmic_BptxAND_MaxPixelTrack_v5',
    'HLT_HIUPC_DoubleMuCosmic_NotMBHF2AND_MaxPixelTrack_v5',
    'HLT_HIUPC_DoubleMuCosmic_NotMBHF2AND_v5',
    'HLT_HIUPC_DoubleMuOpen_BptxAND_MaxPixelTrack_v7',
    'HLT_HIUPC_DoubleMuOpen_NotMBHF2AND_MaxPixelTrack_v5',
    'HLT_HIUPC_DoubleMuOpen_NotMBHF2AND_v5',
    'HLT_HIUPC_MBHF1AND_PixelTrackMultiplicity20400_v5',
    'HLT_HIUPC_MBHF1AND_PixelTrackMultiplicity30400_v5',
    'HLT_HIUPC_MBHF1AND_PixelTrackMultiplicity40400_v5',
    'HLT_HIUPC_MBHF2AND_PixelTrackMultiplicity20400_v5',
    'HLT_HIUPC_MBHF2AND_PixelTrackMultiplicity30400_v5',
    'HLT_HIUPC_MBHF2AND_PixelTrackMultiplicity40400_v5',
    'HLT_HIUPC_SingleEG2_NotMBHF2AND_ZDC1nOR_SinglePixelTrack_MaxPixelTrack_v5',
    'HLT_HIUPC_SingleEG3_BptxAND_SinglePixelTrack_MaxPixelTrack_v7',
    'HLT_HIUPC_SingleEG3_NotMBHF2AND_SinglePixelTrack_MaxPixelTrack_v7',
    'HLT_HIUPC_SingleEG3_NotMBHF2AND_v7',
    'HLT_HIUPC_SingleEG3_NotMBHF2OR_SinglePixelTrack_MaxPixelTrack_v7',
    'HLT_HIUPC_SingleEG3_NotMBHF2OR_v7',
    'HLT_HIUPC_SingleEG5_BptxAND_SinglePixelTrack_MaxPixelTrack_v7',
    'HLT_HIUPC_SingleEG5_NotMBHF2AND_SinglePixelTrack_MaxPixelTrack_v7',
    'HLT_HIUPC_SingleEG5_NotMBHF2AND_v7',
    'HLT_HIUPC_SingleMuCosmic_BptxAND_MaxPixelTrack_v5',
    'HLT_HIUPC_SingleMuCosmic_NotMBHF2AND_MaxPixelTrack_v5',
    'HLT_HIUPC_SingleMuCosmic_NotMBHF2AND_v5',
    'HLT_HIUPC_SingleMuCosmic_NotMBHF2OR_MaxPixelTrack_v5',
    'HLT_HIUPC_SingleMuCosmic_NotMBHF2OR_v5',
    'HLT_HIUPC_SingleMuOpen_BptxAND_MaxPixelTrack_v7',
    'HLT_HIUPC_SingleMuOpen_NotMBHF2AND_MaxPixelTrack_v7',
    'HLT_HIUPC_SingleMuOpen_NotMBHF2AND_v7',
    'HLT_HIUPC_SingleMuOpen_NotMBHF2OR_MaxPixelTrack_v7',
    'HLT_HIUPC_SingleMuOpen_NotMBHF2OR_v7',
    'HLT_HIUPC_SingleMuOpen_OR_SingleMuCosmic_EMTF_BptxAND_MaxPixelTrack_v5',
    'HLT_HIUPC_SingleMuOpen_OR_SingleMuCosmic_EMTF_NotMBHF2AND_MaxPixelTrack_v5',
    'HLT_HIUPC_SingleMuOpen_OR_SingleMuCosmic_EMTF_NotMBHF2AND_v5',
    'HLT_HIUPC_SingleMuOpen_OR_SingleMuCosmic_EMTF_NotMBHF2OR_MaxPixelTrack_v5',
    'HLT_HIUPC_SingleMuOpen_OR_SingleMuCosmic_EMTF_NotMBHF2OR_v5',
    'HLT_HIUPC_ZeroBias_MinPixelCluster400_MaxPixelCluster10000_v5',
    'HLT_HIUPC_ZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v5',
    'HLT_HIUPC_ZeroBias_SinglePixelTrack_MaxPixelTrack_v5',
    'HLT_HIZeroBias_v7'
)


# stream PhysicsHITrackerNZS

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHITrackerNZS_datasetHITrackerNZS_selector
streamPhysicsHITrackerNZS_datasetHITrackerNZS_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHITrackerNZS_datasetHITrackerNZS_selector.l1tResults = cms.InputTag('')
streamPhysicsHITrackerNZS_datasetHITrackerNZS_selector.throw      = cms.bool(False)
streamPhysicsHITrackerNZS_datasetHITrackerNZS_selector.triggerConditions = cms.vstring('HLT_HIPhysicsForZS_v7')

