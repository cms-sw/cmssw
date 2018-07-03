# /dev/CMSSW_10_1_0/PRef

import FWCore.ParameterSet.Config as cms


# stream PhysicsCommissioning

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetHLTPhysics_selector
streamPhysicsCommissioning_datasetHLTPhysics_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetHLTPhysics_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetHLTPhysics_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetHLTPhysics_selector.triggerConditions = cms.vstring('HLT_Physics_v7')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetHighEGJet_selector
streamPhysicsCommissioning_datasetHighEGJet_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetHighEGJet_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetHighEGJet_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetHighEGJet_selector.triggerConditions = cms.vstring(
    'HLT_HIAK4CaloJet100_v1', 
    'HLT_HIAK4CaloJet120_v1', 
    'HLT_HIAK4CaloJet80FWD_v1', 
    'HLT_HIAK4CaloJet80_v1', 
    'HLT_HIAK4PFJet100_v3', 
    'HLT_HIAK4PFJet120_v3', 
    'HLT_HIAK4PFJet30_bTag_v3', 
    'HLT_HIAK4PFJet40_bTag_v3', 
    'HLT_HIAK4PFJet60_bTag_v3', 
    'HLT_HIAK4PFJet80FWD_v3', 
    'HLT_HIAK4PFJet80_bTag_v3', 
    'HLT_HIAK4PFJet80_v3', 
    'HLT_HIAK8PFJet140_v3', 
    'HLT_HIAK8PFJet80_v3', 
    'HLT_HIDoublePhoton15_Eta3p1ForPPRef_Mass50to1000_v8', 
    'HLT_HIEle10_WPLoose_Gsf_v3', 
    'HLT_HIEle15_Ele8_CaloIdL_TrackIdL_IsoVL_v3', 
    'HLT_HIEle15_WPLoose_Gsf_v3', 
    'HLT_HIEle17_WPLoose_Gsf_v3', 
    'HLT_HIEle20_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v3', 
    'HLT_HIEle20_WPLoose_Gsf_v3', 
    'HLT_HIEle20_eta2p1_WPTight_Gsf_CentralPFJet15_EleCleaned_v3', 
    'HLT_HIEle30_WPLoose_Gsf_v3', 
    'HLT_HIEle40_WPLoose_Gsf_v3', 
    'HLT_HIL3Mu5_AK4PFJet30_v3', 
    'HLT_HIL3Mu5_AK4PFJet40_v3', 
    'HLT_HIL3Mu5_AK4PFJet60_v3', 
    'HLT_HIPFJet140_v3', 
    'HLT_HIPhoton40_HoverELoose_v2', 
    'HLT_HIPhoton50_HoverELoose_v2', 
    'HLT_HIPhoton60_HoverELoose_v2', 
    'HLT_HISinglePhoton40_Eta1p5ForPPRef_v8', 
    'HLT_HISinglePhoton40_Eta3p1ForPPRef_v8', 
    'HLT_HISinglePhoton50_Eta1p5ForPPRef_v8', 
    'HLT_HISinglePhoton50_Eta3p1ForPPRef_v8', 
    'HLT_HISinglePhoton60_Eta1p5ForPPRef_v8', 
    'HLT_HISinglePhoton60_Eta3p1ForPPRef_v8'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetLowEGJet_selector
streamPhysicsCommissioning_datasetLowEGJet_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetLowEGJet_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetLowEGJet_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetLowEGJet_selector.triggerConditions = cms.vstring(
    'HLT_HIAK4CaloJet15_v1', 
    'HLT_HIAK4CaloJet30FWD_v1', 
    'HLT_HIAK4CaloJet30_v1', 
    'HLT_HIAK4CaloJet40FWD_v1', 
    'HLT_HIAK4CaloJet40_v1', 
    'HLT_HIAK4CaloJet60FWD_v1', 
    'HLT_HIAK4CaloJet60_v1', 
    'HLT_HIAK4PFJet15_v3', 
    'HLT_HIAK4PFJet30FWD_v3', 
    'HLT_HIAK4PFJet30_v3', 
    'HLT_HIAK4PFJet40FWD_v3', 
    'HLT_HIAK4PFJet40_v3', 
    'HLT_HIAK4PFJet60FWD_v3', 
    'HLT_HIAK4PFJet60_v3', 
    'HLT_HIAK8PFJet15_v3', 
    'HLT_HIAK8PFJet25_v3', 
    'HLT_HIAK8PFJet40_v3', 
    'HLT_HIAK8PFJetFwd15_v3', 
    'HLT_HIAK8PFJetFwd25_v3', 
    'HLT_HIAK8PFJetFwd40_v3', 
    'HLT_HIPFJet25_v3', 
    'HLT_HIPFJetFwd15_v3', 
    'HLT_HIPFJetFwd25_v3', 
    'HLT_HIPhoton20_HoverELoose_v2', 
    'HLT_HIPhoton30_HoverELoose_v2', 
    'HLT_HISinglePhoton10_Eta1p5ForPPRef_v8', 
    'HLT_HISinglePhoton10_Eta3p1ForPPRef_v8', 
    'HLT_HISinglePhoton15_Eta1p5ForPPRef_v8', 
    'HLT_HISinglePhoton15_Eta3p1ForPPRef_v8', 
    'HLT_HISinglePhoton20_Eta1p5ForPPRef_v8', 
    'HLT_HISinglePhoton20_Eta3p1ForPPRef_v8', 
    'HLT_HISinglePhoton30_Eta1p5ForPPRef_v8', 
    'HLT_HISinglePhoton30_Eta3p1ForPPRef_v8'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetZeroBias_selector
streamPhysicsCommissioning_datasetZeroBias_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetZeroBias_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetZeroBias_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetZeroBias_selector.triggerConditions = cms.vstring(
    'HLT_Random_v3', 
    'HLT_ZeroBias_FirstCollisionAfterAbortGap_v5', 
    'HLT_ZeroBias_v6'
)


# stream PhysicsEndOfFill

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsEndOfFill_datasetEmptyBX_selector
streamPhysicsEndOfFill_datasetEmptyBX_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsEndOfFill_datasetEmptyBX_selector.l1tResults = cms.InputTag('')
streamPhysicsEndOfFill_datasetEmptyBX_selector.throw      = cms.bool(False)
streamPhysicsEndOfFill_datasetEmptyBX_selector.triggerConditions = cms.vstring(
    'HLT_HIL1NotBptxOR_v1', 
    'HLT_HIL1UnpairedBunchBptxMinus_v1', 
    'HLT_HIL1UnpairedBunchBptxPlus_v1'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsEndOfFill_datasetFSQJet1_selector
streamPhysicsEndOfFill_datasetFSQJet1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsEndOfFill_datasetFSQJet1_selector.l1tResults = cms.InputTag('')
streamPhysicsEndOfFill_datasetFSQJet1_selector.throw      = cms.bool(False)
streamPhysicsEndOfFill_datasetFSQJet1_selector.triggerConditions = cms.vstring(
    'HLT_HIDiPFJet15_NoCaloMatched_v3', 
    'HLT_HIDiPFJet25_NoCaloMatched_v3'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsEndOfFill_datasetFSQJet2_selector
streamPhysicsEndOfFill_datasetFSQJet2_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsEndOfFill_datasetFSQJet2_selector.l1tResults = cms.InputTag('')
streamPhysicsEndOfFill_datasetFSQJet2_selector.throw      = cms.bool(False)
streamPhysicsEndOfFill_datasetFSQJet2_selector.triggerConditions = cms.vstring(
    'HLT_HIDiPFJet15_FBEta3_NoCaloMatched_v3', 
    'HLT_HIDiPFJet25_FBEta3_NoCaloMatched_v3', 
    'HLT_HIDiPFJetAve15_HFJEC_v3', 
    'HLT_HIDiPFJetAve25_HFJEC_v3', 
    'HLT_HIDiPFJetAve35_HFJEC_v3'
)


# stream PhysicsHIZeroBias1

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIZeroBias1_datasetHIZeroBias1_selector
streamPhysicsHIZeroBias1_datasetHIZeroBias1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIZeroBias1_datasetHIZeroBias1_selector.l1tResults = cms.InputTag('')
streamPhysicsHIZeroBias1_datasetHIZeroBias1_selector.throw      = cms.bool(False)
streamPhysicsHIZeroBias1_datasetHIZeroBias1_selector.triggerConditions = cms.vstring('HLT_HIZeroBias_part0_v6')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIZeroBias1_datasetHIZeroBias2_selector
streamPhysicsHIZeroBias1_datasetHIZeroBias2_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIZeroBias1_datasetHIZeroBias2_selector.l1tResults = cms.InputTag('')
streamPhysicsHIZeroBias1_datasetHIZeroBias2_selector.throw      = cms.bool(False)
streamPhysicsHIZeroBias1_datasetHIZeroBias2_selector.triggerConditions = cms.vstring('HLT_HIZeroBias_part1_v6')


# stream PhysicsHIZeroBias2

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIZeroBias2_datasetHIZeroBias3_selector
streamPhysicsHIZeroBias2_datasetHIZeroBias3_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIZeroBias2_datasetHIZeroBias3_selector.l1tResults = cms.InputTag('')
streamPhysicsHIZeroBias2_datasetHIZeroBias3_selector.throw      = cms.bool(False)
streamPhysicsHIZeroBias2_datasetHIZeroBias3_selector.triggerConditions = cms.vstring('HLT_HIZeroBias_part2_v6')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIZeroBias2_datasetHIZeroBias4_selector
streamPhysicsHIZeroBias2_datasetHIZeroBias4_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIZeroBias2_datasetHIZeroBias4_selector.l1tResults = cms.InputTag('')
streamPhysicsHIZeroBias2_datasetHIZeroBias4_selector.throw      = cms.bool(False)
streamPhysicsHIZeroBias2_datasetHIZeroBias4_selector.triggerConditions = cms.vstring('HLT_HIZeroBias_part3_v6')


# stream PhysicsHIZeroBias3

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIZeroBias3_datasetHIZeroBias5_selector
streamPhysicsHIZeroBias3_datasetHIZeroBias5_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIZeroBias3_datasetHIZeroBias5_selector.l1tResults = cms.InputTag('')
streamPhysicsHIZeroBias3_datasetHIZeroBias5_selector.throw      = cms.bool(False)
streamPhysicsHIZeroBias3_datasetHIZeroBias5_selector.triggerConditions = cms.vstring('HLT_HIZeroBias_part4_v6')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIZeroBias3_datasetHIZeroBias6_selector
streamPhysicsHIZeroBias3_datasetHIZeroBias6_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIZeroBias3_datasetHIZeroBias6_selector.l1tResults = cms.InputTag('')
streamPhysicsHIZeroBias3_datasetHIZeroBias6_selector.throw      = cms.bool(False)
streamPhysicsHIZeroBias3_datasetHIZeroBias6_selector.triggerConditions = cms.vstring('HLT_HIZeroBias_part5_v6')


# stream PhysicsHIZeroBias4

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIZeroBias4_datasetHIZeroBias7_selector
streamPhysicsHIZeroBias4_datasetHIZeroBias7_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIZeroBias4_datasetHIZeroBias7_selector.l1tResults = cms.InputTag('')
streamPhysicsHIZeroBias4_datasetHIZeroBias7_selector.throw      = cms.bool(False)
streamPhysicsHIZeroBias4_datasetHIZeroBias7_selector.triggerConditions = cms.vstring('HLT_HIZeroBias_part6_v6')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIZeroBias4_datasetHIZeroBias8_selector
streamPhysicsHIZeroBias4_datasetHIZeroBias8_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIZeroBias4_datasetHIZeroBias8_selector.l1tResults = cms.InputTag('')
streamPhysicsHIZeroBias4_datasetHIZeroBias8_selector.throw      = cms.bool(False)
streamPhysicsHIZeroBias4_datasetHIZeroBias8_selector.triggerConditions = cms.vstring('HLT_HIZeroBias_part7_v6')


# stream PhysicsHIZeroBias5

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIZeroBias5_datasetHIZeroBias10_selector
streamPhysicsHIZeroBias5_datasetHIZeroBias10_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIZeroBias5_datasetHIZeroBias10_selector.l1tResults = cms.InputTag('')
streamPhysicsHIZeroBias5_datasetHIZeroBias10_selector.throw      = cms.bool(False)
streamPhysicsHIZeroBias5_datasetHIZeroBias10_selector.triggerConditions = cms.vstring('HLT_HIZeroBias_part9_v6')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIZeroBias5_datasetHIZeroBias9_selector
streamPhysicsHIZeroBias5_datasetHIZeroBias9_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIZeroBias5_datasetHIZeroBias9_selector.l1tResults = cms.InputTag('')
streamPhysicsHIZeroBias5_datasetHIZeroBias9_selector.throw      = cms.bool(False)
streamPhysicsHIZeroBias5_datasetHIZeroBias9_selector.triggerConditions = cms.vstring('HLT_HIZeroBias_part8_v6')


# stream PhysicsHIZeroBias6

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIZeroBias6_datasetHIZeroBias11_selector
streamPhysicsHIZeroBias6_datasetHIZeroBias11_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIZeroBias6_datasetHIZeroBias11_selector.l1tResults = cms.InputTag('')
streamPhysicsHIZeroBias6_datasetHIZeroBias11_selector.throw      = cms.bool(False)
streamPhysicsHIZeroBias6_datasetHIZeroBias11_selector.triggerConditions = cms.vstring('HLT_HIZeroBias_part10_v6')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIZeroBias6_datasetHIZeroBias12_selector
streamPhysicsHIZeroBias6_datasetHIZeroBias12_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIZeroBias6_datasetHIZeroBias12_selector.l1tResults = cms.InputTag('')
streamPhysicsHIZeroBias6_datasetHIZeroBias12_selector.throw      = cms.bool(False)
streamPhysicsHIZeroBias6_datasetHIZeroBias12_selector.triggerConditions = cms.vstring('HLT_HIZeroBias_part11_v6')


# stream PhysicsHadronsTaus

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHadronsTaus_datasetHeavyFlavor_selector
streamPhysicsHadronsTaus_datasetHeavyFlavor_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHadronsTaus_datasetHeavyFlavor_selector.l1tResults = cms.InputTag('')
streamPhysicsHadronsTaus_datasetHeavyFlavor_selector.throw      = cms.bool(False)
streamPhysicsHadronsTaus_datasetHeavyFlavor_selector.triggerConditions = cms.vstring(
    'HLT_HIDijet16And12_MidEta2p7_Dpt10_v1', 
    'HLT_HIDijet16And8_MidEta2p7_Dpt8_v1', 
    'HLT_HIDijet20And12_MidEta2p7_Dpt10_v1', 
    'HLT_HIDijet20And8_MidEta2p7_Dpt8_v1', 
    'HLT_HIDijet28And16_MidEta2p7_Dpt15_v1', 
    'HLT_HIDmesonPPTrackingGlobal_Dpt15_v1', 
    'HLT_HIDmesonPPTrackingGlobal_Dpt30_v1', 
    'HLT_HIDmesonPPTrackingGlobal_Dpt40_v1', 
    'HLT_HIDmesonPPTrackingGlobal_Dpt50_v1', 
    'HLT_HIDmesonPPTrackingGlobal_Dpt60_v1', 
    'HLT_HIDmesonPPTrackingGlobal_Dpt8_v1'
)


# stream PhysicsMuons

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsMuons_datasetDoubleMuon_selector
streamPhysicsMuons_datasetDoubleMuon_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsMuons_datasetDoubleMuon_selector.l1tResults = cms.InputTag('')
streamPhysicsMuons_datasetDoubleMuon_selector.throw      = cms.bool(False)
streamPhysicsMuons_datasetDoubleMuon_selector.triggerConditions = cms.vstring(
    'HLT_HIDimuon0_Jpsi_NoVertexing_v2', 
    'HLT_HIDimuon0_Jpsi_v2', 
    'HLT_HIDimuon0_Upsilon_NoVertexing_v2', 
    'HLT_HIL1DoubleMu0_HighQ_v1', 
    'HLT_HIL1DoubleMu0_v1', 
    'HLT_HIL1DoubleMu10_v1', 
    'HLT_HIL1DoubleMuOpen_OS_v1', 
    'HLT_HIL1DoubleMuOpen_SS_v1', 
    'HLT_HIL1DoubleMuOpen_v1', 
    'HLT_HIL2DoubleMu0_v1', 
    'HLT_HIL2DoubleMu10_v1', 
    'HLT_HIL3DoubleMu0_v2', 
    'HLT_HIL3DoubleMu10_v2'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsMuons_datasetSingleMuon_selector
streamPhysicsMuons_datasetSingleMuon_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsMuons_datasetSingleMuon_selector.l1tResults = cms.InputTag('')
streamPhysicsMuons_datasetSingleMuon_selector.throw      = cms.bool(False)
streamPhysicsMuons_datasetSingleMuon_selector.triggerConditions = cms.vstring(
    'HLT_HIL1Mu12_v1', 
    'HLT_HIL1Mu16_v1', 
    'HLT_HIL2Mu12_v1', 
    'HLT_HIL2Mu15_v1', 
    'HLT_HIL2Mu20_v1', 
    'HLT_HIL2Mu3_NHitQ10_v1', 
    'HLT_HIL2Mu5_NHitQ10_v1', 
    'HLT_HIL2Mu7_v1', 
    'HLT_HIL3Mu12_v2', 
    'HLT_HIL3Mu15_v2', 
    'HLT_HIL3Mu20_v2', 
    'HLT_HIL3Mu3_NHitQ10_v2', 
    'HLT_HIL3Mu3_Track1_Jpsi_v2', 
    'HLT_HIL3Mu3_Track1_v2', 
    'HLT_HIL3Mu3_v2', 
    'HLT_HIL3Mu5_NHitQ10_v2', 
    'HLT_HIL3Mu5_Track1_Jpsi_v2', 
    'HLT_HIL3Mu5_Track1_v2', 
    'HLT_HIL3Mu5_v2', 
    'HLT_HIL3Mu7_v2', 
    'HLT_HIMu7p5_L2Mu2_Jpsi_v2', 
    'HLT_HIMu7p5_L2Mu2_Upsilon_v2', 
    'HLT_HIMu7p5_Track2_Jpsi_v2', 
    'HLT_HIMu7p5_Track2_Upsilon_v2'
)


# stream PhysicsTracks

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsTracks_datasetSingleTrack_selector
streamPhysicsTracks_datasetSingleTrack_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsTracks_datasetSingleTrack_selector.l1tResults = cms.InputTag('')
streamPhysicsTracks_datasetSingleTrack_selector.throw      = cms.bool(False)
streamPhysicsTracks_datasetSingleTrack_selector.triggerConditions = cms.vstring(
    'HLT_HIFullTracks_HighPt18_v1', 
    'HLT_HIFullTracks_HighPt24_v1', 
    'HLT_HIFullTracks_HighPt34_v1', 
    'HLT_HIFullTracks_HighPt45_v1'
)

