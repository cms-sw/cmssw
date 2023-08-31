# /dev/CMSSW_13_2_0/PRef

import FWCore.ParameterSet.Config as cms


# stream PhysicsCommissioning

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetCommissioning_selector
streamPhysicsCommissioning_datasetCommissioning_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetCommissioning_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetCommissioning_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetCommissioning_selector.triggerConditions = cms.vstring(
    'HLT_IsoTrackHB_v9',
    'HLT_IsoTrackHE_v9'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetCommissioningZDC_selector
streamPhysicsCommissioning_datasetCommissioningZDC_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetCommissioningZDC_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetCommissioningZDC_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetCommissioningZDC_selector.triggerConditions = cms.vstring('HLT_ZDCCommissioning_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetEmptyBX_selector
streamPhysicsCommissioning_datasetEmptyBX_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetEmptyBX_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetEmptyBX_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetEmptyBX_selector.triggerConditions = cms.vstring(
    'HLT_HIL1NotBptxORForPPRef_v4',
    'HLT_HIL1UnpairedBunchBptxMinusForPPRef_v4',
    'HLT_HIL1UnpairedBunchBptxPlusForPPRef_v4'
)

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

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetNoBPTX_selector
streamPhysicsCommissioning_datasetNoBPTX_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetNoBPTX_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetNoBPTX_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetNoBPTX_selector.triggerConditions = cms.vstring(
    'HLT_CDC_L2cosmic_10_er1p0_v5',
    'HLT_CDC_L2cosmic_5p5_er1p0_v5'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetZeroBias_selector
streamPhysicsCommissioning_datasetZeroBias_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetZeroBias_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetZeroBias_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetZeroBias_selector.triggerConditions = cms.vstring(
    'HLT_Random_v3',
    'HLT_ZeroBias_FirstCollisionAfterAbortGap_v7',
    'HLT_ZeroBias_v8'
)


# stream PhysicsPPRefHardProbes

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefHardProbes_datasetPPRefHardProbes_selector
streamPhysicsPPRefHardProbes_datasetPPRefHardProbes_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefHardProbes_datasetPPRefHardProbes_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefHardProbes_datasetPPRefHardProbes_selector.throw      = cms.bool(False)
streamPhysicsPPRefHardProbes_datasetPPRefHardProbes_selector.triggerConditions = cms.vstring(
    'HLT_AK4CaloJet100_v1',
    'HLT_AK4CaloJet120_v1',
    'HLT_AK4CaloJet40_v1',
    'HLT_AK4CaloJet60_v1',
    'HLT_AK4CaloJet70_v1',
    'HLT_AK4CaloJet80_v1',
    'HLT_AK4CaloJetFwd100_v1',
    'HLT_AK4CaloJetFwd120_v1',
    'HLT_AK4CaloJetFwd40_v1',
    'HLT_AK4CaloJetFwd60_v1',
    'HLT_AK4CaloJetFwd70_v1',
    'HLT_AK4CaloJetFwd80_v1',
    'HLT_AK4PFJet100_v1',
    'HLT_AK4PFJet120_v1',
    'HLT_AK4PFJet40_v1',
    'HLT_AK4PFJet60_v1',
    'HLT_AK4PFJet80_v1',
    'HLT_AK4PFJetFwd100_v1',
    'HLT_AK4PFJetFwd120_v1',
    'HLT_AK4PFJetFwd40_v1',
    'HLT_AK4PFJetFwd60_v1',
    'HLT_AK4PFJetFwd80_v1',
    'HLT_PPRefDmesonTrackingGlobal_Dpt25_v1',
    'HLT_PPRefDmesonTrackingGlobal_Dpt35_v1',
    'HLT_PPRefDmesonTrackingGlobal_Dpt45_v1',
    'HLT_PPRefDmesonTrackingGlobal_Dpt60_v1',
    'HLT_PPRefDoubleEle10GsfMass50_v1',
    'HLT_PPRefDoubleEle10Gsf_v1',
    'HLT_PPRefDoubleEle15GsfMass50_v1',
    'HLT_PPRefDoubleEle15Gsf_v1',
    'HLT_PPRefEle10Gsf_v1',
    'HLT_PPRefEle15Ele10GsfMass50_v1',
    'HLT_PPRefEle15Ele10Gsf_v1',
    'HLT_PPRefEle15Gsf_v1',
    'HLT_PPRefEle20Gsf_v1',
    'HLT_PPRefEle30Gsf_v1',
    'HLT_PPRefEle40Gsf_v1',
    'HLT_PPRefEle50Gsf_v1',
    'HLT_PPRefGEDPhoton10_EB_v1',
    'HLT_PPRefGEDPhoton10_v1',
    'HLT_PPRefGEDPhoton20_EB_v1',
    'HLT_PPRefGEDPhoton20_v1',
    'HLT_PPRefGEDPhoton30_EB_v1',
    'HLT_PPRefGEDPhoton30_v1',
    'HLT_PPRefGEDPhoton40_EB_v1',
    'HLT_PPRefGEDPhoton40_v1',
    'HLT_PPRefGEDPhoton50_EB_v1',
    'HLT_PPRefGEDPhoton50_v1',
    'HLT_PPRefGEDPhoton60_EB_v1',
    'HLT_PPRefGEDPhoton60_v1'
)


# stream PhysicsPPRefMuons

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefMuons_datasetPPRefDoubleMuon_selector
streamPhysicsPPRefMuons_datasetPPRefDoubleMuon_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefMuons_datasetPPRefDoubleMuon_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefMuons_datasetPPRefDoubleMuon_selector.throw      = cms.bool(False)
streamPhysicsPPRefMuons_datasetPPRefDoubleMuon_selector.triggerConditions = cms.vstring(
    'HLT_PPRefL1DoubleMu0_Open_v1',
    'HLT_PPRefL1DoubleMu0_v1',
    'HLT_PPRefL2DoubleMu0_Open_v1',
    'HLT_PPRefL2DoubleMu0_v1',
    'HLT_PPRefL3DoubleMu0_Open_v1',
    'HLT_PPRefL3DoubleMu0_v1'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefMuons_datasetPPRefExotica_selector
streamPhysicsPPRefMuons_datasetPPRefExotica_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefMuons_datasetPPRefExotica_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefMuons_datasetPPRefExotica_selector.throw      = cms.bool(False)
streamPhysicsPPRefMuons_datasetPPRefExotica_selector.triggerConditions = cms.vstring(
    'HLT_PPRefCscCluster_Loose_v1',
    'HLT_PPRefCscCluster_Medium_v1',
    'HLT_PPRefCscCluster_Tight_v1'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefMuons_datasetPPRefSingleMuon_selector
streamPhysicsPPRefMuons_datasetPPRefSingleMuon_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefMuons_datasetPPRefSingleMuon_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefMuons_datasetPPRefSingleMuon_selector.throw      = cms.bool(False)
streamPhysicsPPRefMuons_datasetPPRefSingleMuon_selector.triggerConditions = cms.vstring(
    'HLT_PPRefL1SingleMu0_Cosmics_v1',
    'HLT_PPRefL1SingleMu12_v1',
    'HLT_PPRefL1SingleMu7_v1',
    'HLT_PPRefL2SingleMu12_v1',
    'HLT_PPRefL2SingleMu15_v1',
    'HLT_PPRefL2SingleMu20_v1',
    'HLT_PPRefL2SingleMu7_v1',
    'HLT_PPRefL3SingleMu12_v1',
    'HLT_PPRefL3SingleMu15_v1',
    'HLT_PPRefL3SingleMu20_v1',
    'HLT_PPRefL3SingleMu3_v1',
    'HLT_PPRefL3SingleMu5_v1',
    'HLT_PPRefL3SingleMu7_v1'
)


# stream PhysicsPPRefZeroBias0

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefZeroBias0_datasetPPRefZeroBias0_selector
streamPhysicsPPRefZeroBias0_datasetPPRefZeroBias0_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefZeroBias0_datasetPPRefZeroBias0_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefZeroBias0_datasetPPRefZeroBias0_selector.throw      = cms.bool(False)
streamPhysicsPPRefZeroBias0_datasetPPRefZeroBias0_selector.triggerConditions = cms.vstring('HLT_PPRefZeroBias_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefZeroBias0_datasetPPRefZeroBias1_selector
streamPhysicsPPRefZeroBias0_datasetPPRefZeroBias1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefZeroBias0_datasetPPRefZeroBias1_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefZeroBias0_datasetPPRefZeroBias1_selector.throw      = cms.bool(False)
streamPhysicsPPRefZeroBias0_datasetPPRefZeroBias1_selector.triggerConditions = cms.vstring('HLT_PPRefZeroBias_v1')


# stream PhysicsPPRefZeroBias1

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefZeroBias1_datasetPPRefZeroBias2_selector
streamPhysicsPPRefZeroBias1_datasetPPRefZeroBias2_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefZeroBias1_datasetPPRefZeroBias2_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefZeroBias1_datasetPPRefZeroBias2_selector.throw      = cms.bool(False)
streamPhysicsPPRefZeroBias1_datasetPPRefZeroBias2_selector.triggerConditions = cms.vstring('HLT_PPRefZeroBias_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefZeroBias1_datasetPPRefZeroBias3_selector
streamPhysicsPPRefZeroBias1_datasetPPRefZeroBias3_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefZeroBias1_datasetPPRefZeroBias3_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefZeroBias1_datasetPPRefZeroBias3_selector.throw      = cms.bool(False)
streamPhysicsPPRefZeroBias1_datasetPPRefZeroBias3_selector.triggerConditions = cms.vstring('HLT_PPRefZeroBias_v1')


# stream PhysicsPPRefZeroBias2

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefZeroBias2_datasetPPRefZeroBias4_selector
streamPhysicsPPRefZeroBias2_datasetPPRefZeroBias4_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefZeroBias2_datasetPPRefZeroBias4_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefZeroBias2_datasetPPRefZeroBias4_selector.throw      = cms.bool(False)
streamPhysicsPPRefZeroBias2_datasetPPRefZeroBias4_selector.triggerConditions = cms.vstring('HLT_PPRefZeroBias_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefZeroBias2_datasetPPRefZeroBias5_selector
streamPhysicsPPRefZeroBias2_datasetPPRefZeroBias5_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefZeroBias2_datasetPPRefZeroBias5_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefZeroBias2_datasetPPRefZeroBias5_selector.throw      = cms.bool(False)
streamPhysicsPPRefZeroBias2_datasetPPRefZeroBias5_selector.triggerConditions = cms.vstring('HLT_PPRefZeroBias_v1')


# stream PhysicsPPRefZeroBias3

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefZeroBias3_datasetPPRefZeroBias6_selector
streamPhysicsPPRefZeroBias3_datasetPPRefZeroBias6_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefZeroBias3_datasetPPRefZeroBias6_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefZeroBias3_datasetPPRefZeroBias6_selector.throw      = cms.bool(False)
streamPhysicsPPRefZeroBias3_datasetPPRefZeroBias6_selector.triggerConditions = cms.vstring('HLT_PPRefZeroBias_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefZeroBias3_datasetPPRefZeroBias7_selector
streamPhysicsPPRefZeroBias3_datasetPPRefZeroBias7_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefZeroBias3_datasetPPRefZeroBias7_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefZeroBias3_datasetPPRefZeroBias7_selector.throw      = cms.bool(False)
streamPhysicsPPRefZeroBias3_datasetPPRefZeroBias7_selector.triggerConditions = cms.vstring('HLT_PPRefZeroBias_v1')


# stream PhysicsPPRefZeroBias4

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefZeroBias4_datasetPPRefZeroBias8_selector
streamPhysicsPPRefZeroBias4_datasetPPRefZeroBias8_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefZeroBias4_datasetPPRefZeroBias8_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefZeroBias4_datasetPPRefZeroBias8_selector.throw      = cms.bool(False)
streamPhysicsPPRefZeroBias4_datasetPPRefZeroBias8_selector.triggerConditions = cms.vstring('HLT_PPRefZeroBias_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefZeroBias4_datasetPPRefZeroBias9_selector
streamPhysicsPPRefZeroBias4_datasetPPRefZeroBias9_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefZeroBias4_datasetPPRefZeroBias9_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefZeroBias4_datasetPPRefZeroBias9_selector.throw      = cms.bool(False)
streamPhysicsPPRefZeroBias4_datasetPPRefZeroBias9_selector.triggerConditions = cms.vstring('HLT_PPRefZeroBias_v1')


# stream PhysicsPPRefZeroBias5

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefZeroBias5_datasetPPRefZeroBias10_selector
streamPhysicsPPRefZeroBias5_datasetPPRefZeroBias10_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefZeroBias5_datasetPPRefZeroBias10_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefZeroBias5_datasetPPRefZeroBias10_selector.throw      = cms.bool(False)
streamPhysicsPPRefZeroBias5_datasetPPRefZeroBias10_selector.triggerConditions = cms.vstring('HLT_PPRefZeroBias_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefZeroBias5_datasetPPRefZeroBias11_selector
streamPhysicsPPRefZeroBias5_datasetPPRefZeroBias11_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefZeroBias5_datasetPPRefZeroBias11_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefZeroBias5_datasetPPRefZeroBias11_selector.throw      = cms.bool(False)
streamPhysicsPPRefZeroBias5_datasetPPRefZeroBias11_selector.triggerConditions = cms.vstring('HLT_PPRefZeroBias_v1')


# stream PhysicsPPRefZeroBias6

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefZeroBias6_datasetPPRefZeroBias12_selector
streamPhysicsPPRefZeroBias6_datasetPPRefZeroBias12_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefZeroBias6_datasetPPRefZeroBias12_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefZeroBias6_datasetPPRefZeroBias12_selector.throw      = cms.bool(False)
streamPhysicsPPRefZeroBias6_datasetPPRefZeroBias12_selector.triggerConditions = cms.vstring('HLT_PPRefZeroBias_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefZeroBias6_datasetPPRefZeroBias13_selector
streamPhysicsPPRefZeroBias6_datasetPPRefZeroBias13_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefZeroBias6_datasetPPRefZeroBias13_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefZeroBias6_datasetPPRefZeroBias13_selector.throw      = cms.bool(False)
streamPhysicsPPRefZeroBias6_datasetPPRefZeroBias13_selector.triggerConditions = cms.vstring('HLT_PPRefZeroBias_v1')


# stream PhysicsPPRefZeroBias7

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefZeroBias7_datasetPPRefZeroBias14_selector
streamPhysicsPPRefZeroBias7_datasetPPRefZeroBias14_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefZeroBias7_datasetPPRefZeroBias14_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefZeroBias7_datasetPPRefZeroBias14_selector.throw      = cms.bool(False)
streamPhysicsPPRefZeroBias7_datasetPPRefZeroBias14_selector.triggerConditions = cms.vstring('HLT_PPRefZeroBias_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefZeroBias7_datasetPPRefZeroBias15_selector
streamPhysicsPPRefZeroBias7_datasetPPRefZeroBias15_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefZeroBias7_datasetPPRefZeroBias15_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefZeroBias7_datasetPPRefZeroBias15_selector.throw      = cms.bool(False)
streamPhysicsPPRefZeroBias7_datasetPPRefZeroBias15_selector.triggerConditions = cms.vstring('HLT_PPRefZeroBias_v1')


# stream PhysicsPPRefZeroBias8

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefZeroBias8_datasetPPRefZeroBias16_selector
streamPhysicsPPRefZeroBias8_datasetPPRefZeroBias16_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefZeroBias8_datasetPPRefZeroBias16_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefZeroBias8_datasetPPRefZeroBias16_selector.throw      = cms.bool(False)
streamPhysicsPPRefZeroBias8_datasetPPRefZeroBias16_selector.triggerConditions = cms.vstring('HLT_PPRefZeroBias_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefZeroBias8_datasetPPRefZeroBias17_selector
streamPhysicsPPRefZeroBias8_datasetPPRefZeroBias17_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefZeroBias8_datasetPPRefZeroBias17_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefZeroBias8_datasetPPRefZeroBias17_selector.throw      = cms.bool(False)
streamPhysicsPPRefZeroBias8_datasetPPRefZeroBias17_selector.triggerConditions = cms.vstring('HLT_PPRefZeroBias_v1')


# stream PhysicsPPRefZeroBias9

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefZeroBias9_datasetPPRefZeroBias18_selector
streamPhysicsPPRefZeroBias9_datasetPPRefZeroBias18_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefZeroBias9_datasetPPRefZeroBias18_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefZeroBias9_datasetPPRefZeroBias18_selector.throw      = cms.bool(False)
streamPhysicsPPRefZeroBias9_datasetPPRefZeroBias18_selector.triggerConditions = cms.vstring('HLT_PPRefZeroBias_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefZeroBias9_datasetPPRefZeroBias19_selector
streamPhysicsPPRefZeroBias9_datasetPPRefZeroBias19_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefZeroBias9_datasetPPRefZeroBias19_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefZeroBias9_datasetPPRefZeroBias19_selector.throw      = cms.bool(False)
streamPhysicsPPRefZeroBias9_datasetPPRefZeroBias19_selector.triggerConditions = cms.vstring('HLT_PPRefZeroBias_v1')

