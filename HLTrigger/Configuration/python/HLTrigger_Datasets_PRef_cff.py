# /dev/CMSSW_14_1_0/PRef

import FWCore.ParameterSet.Config as cms


# stream PhysicsCommissioning

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetCommissioning_selector
streamPhysicsCommissioning_datasetCommissioning_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetCommissioning_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetCommissioning_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetCommissioning_selector.triggerConditions = cms.vstring(
    'HLT_IsoTrackHB_v14',
    'HLT_IsoTrackHE_v14'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetEmptyBX_selector
streamPhysicsCommissioning_datasetEmptyBX_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetEmptyBX_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetEmptyBX_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetEmptyBX_selector.triggerConditions = cms.vstring(
    'HLT_HIL1NotBptxORForPPRef_v9',
    'HLT_HIL1UnpairedBunchBptxMinusForPPRef_v9',
    'HLT_HIL1UnpairedBunchBptxPlusForPPRef_v9'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetHLTPhysics_selector
streamPhysicsCommissioning_datasetHLTPhysics_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetHLTPhysics_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetHLTPhysics_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetHLTPhysics_selector.triggerConditions = cms.vstring('HLT_Physics_v14')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetHcalNZS_selector
streamPhysicsCommissioning_datasetHcalNZS_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetHcalNZS_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetHcalNZS_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetHcalNZS_selector.triggerConditions = cms.vstring(
    'HLT_HcalNZS_v21',
    'HLT_HcalPhiSym_v23'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetNoBPTX_selector
streamPhysicsCommissioning_datasetNoBPTX_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetNoBPTX_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetNoBPTX_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetNoBPTX_selector.triggerConditions = cms.vstring(
    'HLT_CDC_L2cosmic_10_er1p0_v10',
    'HLT_CDC_L2cosmic_5p5_er1p0_v10'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetZeroBias_selector
streamPhysicsCommissioning_datasetZeroBias_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetZeroBias_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetZeroBias_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetZeroBias_selector.triggerConditions = cms.vstring(
    'HLT_Random_v3',
    'HLT_ZeroBias_FirstCollisionAfterAbortGap_v12',
    'HLT_ZeroBias_v13'
)


# stream PhysicsPPRefDoubleMuon0

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefDoubleMuon0_datasetPPRefDoubleMuon0_selector
streamPhysicsPPRefDoubleMuon0_datasetPPRefDoubleMuon0_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefDoubleMuon0_datasetPPRefDoubleMuon0_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefDoubleMuon0_datasetPPRefDoubleMuon0_selector.throw      = cms.bool(False)
streamPhysicsPPRefDoubleMuon0_datasetPPRefDoubleMuon0_selector.triggerConditions = cms.vstring(
    'HLT_PPRefL1DoubleMu0_Open_v6',
    'HLT_PPRefL1DoubleMu0_SQ_v1',
    'HLT_PPRefL1DoubleMu0_v6',
    'HLT_PPRefL1DoubleMu2_SQ_v1',
    'HLT_PPRefL1DoubleMu2_v1',
    'HLT_PPRefL2DoubleMu0_Open_v6',
    'HLT_PPRefL2DoubleMu0_v6',
    'HLT_PPRefL3DoubleMu0_Open_v8',
    'HLT_PPRefL3DoubleMu0_v8'
)


# stream PhysicsPPRefDoubleMuon1

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefDoubleMuon1_datasetPPRefDoubleMuon1_selector
streamPhysicsPPRefDoubleMuon1_datasetPPRefDoubleMuon1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefDoubleMuon1_datasetPPRefDoubleMuon1_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefDoubleMuon1_datasetPPRefDoubleMuon1_selector.throw      = cms.bool(False)
streamPhysicsPPRefDoubleMuon1_datasetPPRefDoubleMuon1_selector.triggerConditions = cms.vstring(
    'HLT_PPRefL1DoubleMu0_Open_v6',
    'HLT_PPRefL1DoubleMu0_SQ_v1',
    'HLT_PPRefL1DoubleMu0_v6',
    'HLT_PPRefL1DoubleMu2_SQ_v1',
    'HLT_PPRefL1DoubleMu2_v1',
    'HLT_PPRefL2DoubleMu0_Open_v6',
    'HLT_PPRefL2DoubleMu0_v6',
    'HLT_PPRefL3DoubleMu0_Open_v8',
    'HLT_PPRefL3DoubleMu0_v8'
)


# stream PhysicsPPRefDoubleMuon2

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefDoubleMuon2_datasetPPRefDoubleMuon2_selector
streamPhysicsPPRefDoubleMuon2_datasetPPRefDoubleMuon2_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefDoubleMuon2_datasetPPRefDoubleMuon2_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefDoubleMuon2_datasetPPRefDoubleMuon2_selector.throw      = cms.bool(False)
streamPhysicsPPRefDoubleMuon2_datasetPPRefDoubleMuon2_selector.triggerConditions = cms.vstring(
    'HLT_PPRefL1DoubleMu0_Open_v6',
    'HLT_PPRefL1DoubleMu0_SQ_v1',
    'HLT_PPRefL1DoubleMu0_v6',
    'HLT_PPRefL1DoubleMu2_SQ_v1',
    'HLT_PPRefL1DoubleMu2_v1',
    'HLT_PPRefL2DoubleMu0_Open_v6',
    'HLT_PPRefL2DoubleMu0_v6',
    'HLT_PPRefL3DoubleMu0_Open_v8',
    'HLT_PPRefL3DoubleMu0_v8'
)


# stream PhysicsPPRefDoubleMuon3

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefDoubleMuon3_datasetPPRefDoubleMuon3_selector
streamPhysicsPPRefDoubleMuon3_datasetPPRefDoubleMuon3_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefDoubleMuon3_datasetPPRefDoubleMuon3_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefDoubleMuon3_datasetPPRefDoubleMuon3_selector.throw      = cms.bool(False)
streamPhysicsPPRefDoubleMuon3_datasetPPRefDoubleMuon3_selector.triggerConditions = cms.vstring(
    'HLT_PPRefL1DoubleMu0_Open_v6',
    'HLT_PPRefL1DoubleMu0_SQ_v1',
    'HLT_PPRefL1DoubleMu0_v6',
    'HLT_PPRefL1DoubleMu2_SQ_v1',
    'HLT_PPRefL1DoubleMu2_v1',
    'HLT_PPRefL2DoubleMu0_Open_v6',
    'HLT_PPRefL2DoubleMu0_v6',
    'HLT_PPRefL3DoubleMu0_Open_v8',
    'HLT_PPRefL3DoubleMu0_v8'
)


# stream PhysicsPPRefHardProbes0

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefHardProbes0_datasetPPRefHardProbes0_selector
streamPhysicsPPRefHardProbes0_datasetPPRefHardProbes0_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefHardProbes0_datasetPPRefHardProbes0_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefHardProbes0_datasetPPRefHardProbes0_selector.throw      = cms.bool(False)
streamPhysicsPPRefHardProbes0_datasetPPRefHardProbes0_selector.triggerConditions = cms.vstring(
    'HLT_AK4CaloJet100_v6',
    'HLT_AK4CaloJet120_v6',
    'HLT_AK4CaloJet40_v6',
    'HLT_AK4CaloJet60_v6',
    'HLT_AK4CaloJet70_v6',
    'HLT_AK4CaloJet80_v6',
    'HLT_AK4CaloJetFwd100_v6',
    'HLT_AK4CaloJetFwd120_v6',
    'HLT_AK4CaloJetFwd40_v6',
    'HLT_AK4CaloJetFwd60_v6',
    'HLT_AK4CaloJetFwd70_v6',
    'HLT_AK4CaloJetFwd80_v6',
    'HLT_AK4PFJet100_v8',
    'HLT_AK4PFJet120_v8',
    'HLT_AK4PFJet40_v8',
    'HLT_AK4PFJet60_v8',
    'HLT_AK4PFJet80_v8',
    'HLT_AK4PFJetFwd100_v8',
    'HLT_AK4PFJetFwd120_v8',
    'HLT_AK4PFJetFwd40_v8',
    'HLT_AK4PFJetFwd60_v8',
    'HLT_AK4PFJetFwd80_v8',
    'HLT_PPRefDoubleEle10GsfMass50_v6',
    'HLT_PPRefDoubleEle10Gsf_v6',
    'HLT_PPRefDoubleEle15GsfMass50_v6',
    'HLT_PPRefDoubleEle15Gsf_v6',
    'HLT_PPRefDoubleGEDPhoton20_v1',
    'HLT_PPRefEle10Gsf_v6',
    'HLT_PPRefEle15Ele10GsfMass50_v6',
    'HLT_PPRefEle15Ele10Gsf_v6',
    'HLT_PPRefEle15Gsf_v6',
    'HLT_PPRefEle20Gsf_v6',
    'HLT_PPRefEle30Gsf_v6',
    'HLT_PPRefEle40Gsf_v6',
    'HLT_PPRefEle50Gsf_v6',
    'HLT_PPRefGEDPhoton10_EB_v6',
    'HLT_PPRefGEDPhoton10_v6',
    'HLT_PPRefGEDPhoton20_EB_v6',
    'HLT_PPRefGEDPhoton20_v6',
    'HLT_PPRefGEDPhoton30_EB_v6',
    'HLT_PPRefGEDPhoton30_v6',
    'HLT_PPRefGEDPhoton40_EB_v6',
    'HLT_PPRefGEDPhoton40_v6',
    'HLT_PPRefGEDPhoton50_EB_v6',
    'HLT_PPRefGEDPhoton50_v6',
    'HLT_PPRefGEDPhoton60_EB_v6',
    'HLT_PPRefGEDPhoton60_v6'
)


# stream PhysicsPPRefHardProbes1

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefHardProbes1_datasetPPRefHardProbes1_selector
streamPhysicsPPRefHardProbes1_datasetPPRefHardProbes1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefHardProbes1_datasetPPRefHardProbes1_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefHardProbes1_datasetPPRefHardProbes1_selector.throw      = cms.bool(False)
streamPhysicsPPRefHardProbes1_datasetPPRefHardProbes1_selector.triggerConditions = cms.vstring(
    'HLT_AK4CaloJet100_v6',
    'HLT_AK4CaloJet120_v6',
    'HLT_AK4CaloJet40_v6',
    'HLT_AK4CaloJet60_v6',
    'HLT_AK4CaloJet70_v6',
    'HLT_AK4CaloJet80_v6',
    'HLT_AK4CaloJetFwd100_v6',
    'HLT_AK4CaloJetFwd120_v6',
    'HLT_AK4CaloJetFwd40_v6',
    'HLT_AK4CaloJetFwd60_v6',
    'HLT_AK4CaloJetFwd70_v6',
    'HLT_AK4CaloJetFwd80_v6',
    'HLT_AK4PFJet100_v8',
    'HLT_AK4PFJet120_v8',
    'HLT_AK4PFJet40_v8',
    'HLT_AK4PFJet60_v8',
    'HLT_AK4PFJet80_v8',
    'HLT_AK4PFJetFwd100_v8',
    'HLT_AK4PFJetFwd120_v8',
    'HLT_AK4PFJetFwd40_v8',
    'HLT_AK4PFJetFwd60_v8',
    'HLT_AK4PFJetFwd80_v8',
    'HLT_PPRefDoubleEle10GsfMass50_v6',
    'HLT_PPRefDoubleEle10Gsf_v6',
    'HLT_PPRefDoubleEle15GsfMass50_v6',
    'HLT_PPRefDoubleEle15Gsf_v6',
    'HLT_PPRefDoubleGEDPhoton20_v1',
    'HLT_PPRefEle10Gsf_v6',
    'HLT_PPRefEle15Ele10GsfMass50_v6',
    'HLT_PPRefEle15Ele10Gsf_v6',
    'HLT_PPRefEle15Gsf_v6',
    'HLT_PPRefEle20Gsf_v6',
    'HLT_PPRefEle30Gsf_v6',
    'HLT_PPRefEle40Gsf_v6',
    'HLT_PPRefEle50Gsf_v6',
    'HLT_PPRefGEDPhoton10_EB_v6',
    'HLT_PPRefGEDPhoton10_v6',
    'HLT_PPRefGEDPhoton20_EB_v6',
    'HLT_PPRefGEDPhoton20_v6',
    'HLT_PPRefGEDPhoton30_EB_v6',
    'HLT_PPRefGEDPhoton30_v6',
    'HLT_PPRefGEDPhoton40_EB_v6',
    'HLT_PPRefGEDPhoton40_v6',
    'HLT_PPRefGEDPhoton50_EB_v6',
    'HLT_PPRefGEDPhoton50_v6',
    'HLT_PPRefGEDPhoton60_EB_v6',
    'HLT_PPRefGEDPhoton60_v6'
)


# stream PhysicsPPRefHardProbes2

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefHardProbes2_datasetPPRefHardProbes2_selector
streamPhysicsPPRefHardProbes2_datasetPPRefHardProbes2_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefHardProbes2_datasetPPRefHardProbes2_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefHardProbes2_datasetPPRefHardProbes2_selector.throw      = cms.bool(False)
streamPhysicsPPRefHardProbes2_datasetPPRefHardProbes2_selector.triggerConditions = cms.vstring(
    'HLT_AK4CaloJet100_v6',
    'HLT_AK4CaloJet120_v6',
    'HLT_AK4CaloJet40_v6',
    'HLT_AK4CaloJet60_v6',
    'HLT_AK4CaloJet70_v6',
    'HLT_AK4CaloJet80_v6',
    'HLT_AK4CaloJetFwd100_v6',
    'HLT_AK4CaloJetFwd120_v6',
    'HLT_AK4CaloJetFwd40_v6',
    'HLT_AK4CaloJetFwd60_v6',
    'HLT_AK4CaloJetFwd70_v6',
    'HLT_AK4CaloJetFwd80_v6',
    'HLT_AK4PFJet100_v8',
    'HLT_AK4PFJet120_v8',
    'HLT_AK4PFJet40_v8',
    'HLT_AK4PFJet60_v8',
    'HLT_AK4PFJet80_v8',
    'HLT_AK4PFJetFwd100_v8',
    'HLT_AK4PFJetFwd120_v8',
    'HLT_AK4PFJetFwd40_v8',
    'HLT_AK4PFJetFwd60_v8',
    'HLT_AK4PFJetFwd80_v8',
    'HLT_PPRefDoubleEle10GsfMass50_v6',
    'HLT_PPRefDoubleEle10Gsf_v6',
    'HLT_PPRefDoubleEle15GsfMass50_v6',
    'HLT_PPRefDoubleEle15Gsf_v6',
    'HLT_PPRefDoubleGEDPhoton20_v1',
    'HLT_PPRefEle10Gsf_v6',
    'HLT_PPRefEle15Ele10GsfMass50_v6',
    'HLT_PPRefEle15Ele10Gsf_v6',
    'HLT_PPRefEle15Gsf_v6',
    'HLT_PPRefEle20Gsf_v6',
    'HLT_PPRefEle30Gsf_v6',
    'HLT_PPRefEle40Gsf_v6',
    'HLT_PPRefEle50Gsf_v6',
    'HLT_PPRefGEDPhoton10_EB_v6',
    'HLT_PPRefGEDPhoton10_v6',
    'HLT_PPRefGEDPhoton20_EB_v6',
    'HLT_PPRefGEDPhoton20_v6',
    'HLT_PPRefGEDPhoton30_EB_v6',
    'HLT_PPRefGEDPhoton30_v6',
    'HLT_PPRefGEDPhoton40_EB_v6',
    'HLT_PPRefGEDPhoton40_v6',
    'HLT_PPRefGEDPhoton50_EB_v6',
    'HLT_PPRefGEDPhoton50_v6',
    'HLT_PPRefGEDPhoton60_EB_v6',
    'HLT_PPRefGEDPhoton60_v6'
)


# stream PhysicsPPRefSingleMuon0

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefSingleMuon0_datasetPPRefSingleMuon0_selector
streamPhysicsPPRefSingleMuon0_datasetPPRefSingleMuon0_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefSingleMuon0_datasetPPRefSingleMuon0_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefSingleMuon0_datasetPPRefSingleMuon0_selector.throw      = cms.bool(False)
streamPhysicsPPRefSingleMuon0_datasetPPRefSingleMuon0_selector.triggerConditions = cms.vstring(
    'HLT_PPRefCscCluster_Loose_v6',
    'HLT_PPRefCscCluster_Medium_v6',
    'HLT_PPRefCscCluster_Tight_v6',
    'HLT_PPRefL1SingleMu0_Cosmics_v6',
    'HLT_PPRefL1SingleMu12_v6',
    'HLT_PPRefL1SingleMu5_Ele20Gsf_v1',
    'HLT_PPRefL1SingleMu5_GEDPhoton20_v1',
    'HLT_PPRefL1SingleMu7_Ele20Gsf_v1',
    'HLT_PPRefL1SingleMu7_GEDPhoton10_v1',
    'HLT_PPRefL1SingleMu7_v6',
    'HLT_PPRefL2SingleMu12_v6',
    'HLT_PPRefL2SingleMu15_v6',
    'HLT_PPRefL2SingleMu20_v6',
    'HLT_PPRefL2SingleMu7_v6',
    'HLT_PPRefL3SingleMu12_v8',
    'HLT_PPRefL3SingleMu15_v8',
    'HLT_PPRefL3SingleMu20_v8',
    'HLT_PPRefL3SingleMu3_SingleAK4CaloJet40_v1',
    'HLT_PPRefL3SingleMu3_SingleAK4CaloJet60_v1',
    'HLT_PPRefL3SingleMu3_SingleAK4CaloJet80_v1',
    'HLT_PPRefL3SingleMu3_v8',
    'HLT_PPRefL3SingleMu5_SingleAK4CaloJet40_v1',
    'HLT_PPRefL3SingleMu5_SingleAK4CaloJet60_v1',
    'HLT_PPRefL3SingleMu5_SingleAK4CaloJet80_v1',
    'HLT_PPRefL3SingleMu5_v8',
    'HLT_PPRefL3SingleMu7_v8'
)


# stream PhysicsPPRefSingleMuon1

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefSingleMuon1_datasetPPRefSingleMuon1_selector
streamPhysicsPPRefSingleMuon1_datasetPPRefSingleMuon1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefSingleMuon1_datasetPPRefSingleMuon1_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefSingleMuon1_datasetPPRefSingleMuon1_selector.throw      = cms.bool(False)
streamPhysicsPPRefSingleMuon1_datasetPPRefSingleMuon1_selector.triggerConditions = cms.vstring(
    'HLT_PPRefCscCluster_Loose_v6',
    'HLT_PPRefCscCluster_Medium_v6',
    'HLT_PPRefCscCluster_Tight_v6',
    'HLT_PPRefL1SingleMu0_Cosmics_v6',
    'HLT_PPRefL1SingleMu12_v6',
    'HLT_PPRefL1SingleMu5_Ele20Gsf_v1',
    'HLT_PPRefL1SingleMu5_GEDPhoton20_v1',
    'HLT_PPRefL1SingleMu7_Ele20Gsf_v1',
    'HLT_PPRefL1SingleMu7_GEDPhoton10_v1',
    'HLT_PPRefL1SingleMu7_v6',
    'HLT_PPRefL2SingleMu12_v6',
    'HLT_PPRefL2SingleMu15_v6',
    'HLT_PPRefL2SingleMu20_v6',
    'HLT_PPRefL2SingleMu7_v6',
    'HLT_PPRefL3SingleMu12_v8',
    'HLT_PPRefL3SingleMu15_v8',
    'HLT_PPRefL3SingleMu20_v8',
    'HLT_PPRefL3SingleMu3_SingleAK4CaloJet40_v1',
    'HLT_PPRefL3SingleMu3_SingleAK4CaloJet60_v1',
    'HLT_PPRefL3SingleMu3_SingleAK4CaloJet80_v1',
    'HLT_PPRefL3SingleMu3_v8',
    'HLT_PPRefL3SingleMu5_SingleAK4CaloJet40_v1',
    'HLT_PPRefL3SingleMu5_SingleAK4CaloJet60_v1',
    'HLT_PPRefL3SingleMu5_SingleAK4CaloJet80_v1',
    'HLT_PPRefL3SingleMu5_v8',
    'HLT_PPRefL3SingleMu7_v8'
)


# stream PhysicsPPRefSingleMuon2

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefSingleMuon2_datasetPPRefSingleMuon2_selector
streamPhysicsPPRefSingleMuon2_datasetPPRefSingleMuon2_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefSingleMuon2_datasetPPRefSingleMuon2_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefSingleMuon2_datasetPPRefSingleMuon2_selector.throw      = cms.bool(False)
streamPhysicsPPRefSingleMuon2_datasetPPRefSingleMuon2_selector.triggerConditions = cms.vstring(
    'HLT_PPRefCscCluster_Loose_v6',
    'HLT_PPRefCscCluster_Medium_v6',
    'HLT_PPRefCscCluster_Tight_v6',
    'HLT_PPRefL1SingleMu0_Cosmics_v6',
    'HLT_PPRefL1SingleMu12_v6',
    'HLT_PPRefL1SingleMu5_Ele20Gsf_v1',
    'HLT_PPRefL1SingleMu5_GEDPhoton20_v1',
    'HLT_PPRefL1SingleMu7_Ele20Gsf_v1',
    'HLT_PPRefL1SingleMu7_GEDPhoton10_v1',
    'HLT_PPRefL1SingleMu7_v6',
    'HLT_PPRefL2SingleMu12_v6',
    'HLT_PPRefL2SingleMu15_v6',
    'HLT_PPRefL2SingleMu20_v6',
    'HLT_PPRefL2SingleMu7_v6',
    'HLT_PPRefL3SingleMu12_v8',
    'HLT_PPRefL3SingleMu15_v8',
    'HLT_PPRefL3SingleMu20_v8',
    'HLT_PPRefL3SingleMu3_SingleAK4CaloJet40_v1',
    'HLT_PPRefL3SingleMu3_SingleAK4CaloJet60_v1',
    'HLT_PPRefL3SingleMu3_SingleAK4CaloJet80_v1',
    'HLT_PPRefL3SingleMu3_v8',
    'HLT_PPRefL3SingleMu5_SingleAK4CaloJet40_v1',
    'HLT_PPRefL3SingleMu5_SingleAK4CaloJet60_v1',
    'HLT_PPRefL3SingleMu5_SingleAK4CaloJet80_v1',
    'HLT_PPRefL3SingleMu5_v8',
    'HLT_PPRefL3SingleMu7_v8'
)


# stream PhysicsPPRefZeroBiasPlusForward0

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefZeroBiasPlusForward0_datasetPPRefZeroBiasPlusForward0_selector
streamPhysicsPPRefZeroBiasPlusForward0_datasetPPRefZeroBiasPlusForward0_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefZeroBiasPlusForward0_datasetPPRefZeroBiasPlusForward0_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefZeroBiasPlusForward0_datasetPPRefZeroBiasPlusForward0_selector.throw      = cms.bool(False)
streamPhysicsPPRefZeroBiasPlusForward0_datasetPPRefZeroBiasPlusForward0_selector.triggerConditions = cms.vstring(
    'HLT_PPRefUPC_SingleJet12_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet12_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet16_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet16_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet20_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet20_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet24_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet24_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet28_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet28_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet8_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet8_ZDC1nOR_v1',
    'HLT_PPRefUPC_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_ZDC1nOR_v1',
    'HLT_PPRefZeroBias_v6'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefZeroBiasPlusForward0_datasetPPRefZeroBiasPlusForward1_selector
streamPhysicsPPRefZeroBiasPlusForward0_datasetPPRefZeroBiasPlusForward1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefZeroBiasPlusForward0_datasetPPRefZeroBiasPlusForward1_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefZeroBiasPlusForward0_datasetPPRefZeroBiasPlusForward1_selector.throw      = cms.bool(False)
streamPhysicsPPRefZeroBiasPlusForward0_datasetPPRefZeroBiasPlusForward1_selector.triggerConditions = cms.vstring(
    'HLT_PPRefUPC_SingleJet12_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet12_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet16_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet16_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet20_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet20_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet24_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet24_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet28_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet28_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet8_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet8_ZDC1nOR_v1',
    'HLT_PPRefUPC_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_ZDC1nOR_v1',
    'HLT_PPRefZeroBias_v6'
)


# stream PhysicsPPRefZeroBiasPlusForward1

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefZeroBiasPlusForward1_datasetPPRefZeroBiasPlusForward2_selector
streamPhysicsPPRefZeroBiasPlusForward1_datasetPPRefZeroBiasPlusForward2_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefZeroBiasPlusForward1_datasetPPRefZeroBiasPlusForward2_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefZeroBiasPlusForward1_datasetPPRefZeroBiasPlusForward2_selector.throw      = cms.bool(False)
streamPhysicsPPRefZeroBiasPlusForward1_datasetPPRefZeroBiasPlusForward2_selector.triggerConditions = cms.vstring(
    'HLT_PPRefUPC_SingleJet12_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet12_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet16_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet16_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet20_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet20_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet24_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet24_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet28_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet28_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet8_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet8_ZDC1nOR_v1',
    'HLT_PPRefUPC_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_ZDC1nOR_v1',
    'HLT_PPRefZeroBias_v6'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefZeroBiasPlusForward1_datasetPPRefZeroBiasPlusForward3_selector
streamPhysicsPPRefZeroBiasPlusForward1_datasetPPRefZeroBiasPlusForward3_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefZeroBiasPlusForward1_datasetPPRefZeroBiasPlusForward3_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefZeroBiasPlusForward1_datasetPPRefZeroBiasPlusForward3_selector.throw      = cms.bool(False)
streamPhysicsPPRefZeroBiasPlusForward1_datasetPPRefZeroBiasPlusForward3_selector.triggerConditions = cms.vstring(
    'HLT_PPRefUPC_SingleJet12_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet12_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet16_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet16_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet20_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet20_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet24_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet24_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet28_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet28_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet8_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet8_ZDC1nOR_v1',
    'HLT_PPRefUPC_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_ZDC1nOR_v1',
    'HLT_PPRefZeroBias_v6'
)


# stream PhysicsPPRefZeroBiasPlusForward2

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefZeroBiasPlusForward2_datasetPPRefZeroBiasPlusForward4_selector
streamPhysicsPPRefZeroBiasPlusForward2_datasetPPRefZeroBiasPlusForward4_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefZeroBiasPlusForward2_datasetPPRefZeroBiasPlusForward4_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefZeroBiasPlusForward2_datasetPPRefZeroBiasPlusForward4_selector.throw      = cms.bool(False)
streamPhysicsPPRefZeroBiasPlusForward2_datasetPPRefZeroBiasPlusForward4_selector.triggerConditions = cms.vstring(
    'HLT_PPRefUPC_SingleJet12_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet12_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet16_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet16_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet20_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet20_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet24_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet24_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet28_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet28_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet8_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet8_ZDC1nOR_v1',
    'HLT_PPRefUPC_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_ZDC1nOR_v1',
    'HLT_PPRefZeroBias_v6'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefZeroBiasPlusForward2_datasetPPRefZeroBiasPlusForward5_selector
streamPhysicsPPRefZeroBiasPlusForward2_datasetPPRefZeroBiasPlusForward5_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefZeroBiasPlusForward2_datasetPPRefZeroBiasPlusForward5_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefZeroBiasPlusForward2_datasetPPRefZeroBiasPlusForward5_selector.throw      = cms.bool(False)
streamPhysicsPPRefZeroBiasPlusForward2_datasetPPRefZeroBiasPlusForward5_selector.triggerConditions = cms.vstring(
    'HLT_PPRefUPC_SingleJet12_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet12_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet16_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet16_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet20_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet20_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet24_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet24_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet28_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet28_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet8_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet8_ZDC1nOR_v1',
    'HLT_PPRefUPC_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_ZDC1nOR_v1',
    'HLT_PPRefZeroBias_v6'
)


# stream PhysicsPPRefZeroBiasPlusForward3

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefZeroBiasPlusForward3_datasetPPRefZeroBiasPlusForward6_selector
streamPhysicsPPRefZeroBiasPlusForward3_datasetPPRefZeroBiasPlusForward6_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefZeroBiasPlusForward3_datasetPPRefZeroBiasPlusForward6_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefZeroBiasPlusForward3_datasetPPRefZeroBiasPlusForward6_selector.throw      = cms.bool(False)
streamPhysicsPPRefZeroBiasPlusForward3_datasetPPRefZeroBiasPlusForward6_selector.triggerConditions = cms.vstring(
    'HLT_PPRefUPC_SingleJet12_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet12_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet16_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet16_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet20_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet20_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet24_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet24_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet28_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet28_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet8_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet8_ZDC1nOR_v1',
    'HLT_PPRefUPC_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_ZDC1nOR_v1',
    'HLT_PPRefZeroBias_v6'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefZeroBiasPlusForward3_datasetPPRefZeroBiasPlusForward7_selector
streamPhysicsPPRefZeroBiasPlusForward3_datasetPPRefZeroBiasPlusForward7_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefZeroBiasPlusForward3_datasetPPRefZeroBiasPlusForward7_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefZeroBiasPlusForward3_datasetPPRefZeroBiasPlusForward7_selector.throw      = cms.bool(False)
streamPhysicsPPRefZeroBiasPlusForward3_datasetPPRefZeroBiasPlusForward7_selector.triggerConditions = cms.vstring(
    'HLT_PPRefUPC_SingleJet12_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet12_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet16_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet16_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet20_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet20_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet24_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet24_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet28_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet28_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet8_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet8_ZDC1nOR_v1',
    'HLT_PPRefUPC_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_ZDC1nOR_v1',
    'HLT_PPRefZeroBias_v6'
)


# stream PhysicsPPRefZeroBiasPlusForward4

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefZeroBiasPlusForward4_datasetPPRefZeroBiasPlusForward8_selector
streamPhysicsPPRefZeroBiasPlusForward4_datasetPPRefZeroBiasPlusForward8_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefZeroBiasPlusForward4_datasetPPRefZeroBiasPlusForward8_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefZeroBiasPlusForward4_datasetPPRefZeroBiasPlusForward8_selector.throw      = cms.bool(False)
streamPhysicsPPRefZeroBiasPlusForward4_datasetPPRefZeroBiasPlusForward8_selector.triggerConditions = cms.vstring(
    'HLT_PPRefUPC_SingleJet12_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet12_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet16_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet16_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet20_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet20_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet24_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet24_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet28_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet28_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet8_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet8_ZDC1nOR_v1',
    'HLT_PPRefUPC_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_ZDC1nOR_v1',
    'HLT_PPRefZeroBias_v6'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefZeroBiasPlusForward4_datasetPPRefZeroBiasPlusForward9_selector
streamPhysicsPPRefZeroBiasPlusForward4_datasetPPRefZeroBiasPlusForward9_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefZeroBiasPlusForward4_datasetPPRefZeroBiasPlusForward9_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefZeroBiasPlusForward4_datasetPPRefZeroBiasPlusForward9_selector.throw      = cms.bool(False)
streamPhysicsPPRefZeroBiasPlusForward4_datasetPPRefZeroBiasPlusForward9_selector.triggerConditions = cms.vstring(
    'HLT_PPRefUPC_SingleJet12_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet12_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet16_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet16_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet20_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet20_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet24_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet24_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet28_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet28_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet8_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet8_ZDC1nOR_v1',
    'HLT_PPRefUPC_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_ZDC1nOR_v1',
    'HLT_PPRefZeroBias_v6'
)


# stream PhysicsPPRefZeroBiasPlusForward5

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefZeroBiasPlusForward5_datasetPPRefZeroBiasPlusForward10_selector
streamPhysicsPPRefZeroBiasPlusForward5_datasetPPRefZeroBiasPlusForward10_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefZeroBiasPlusForward5_datasetPPRefZeroBiasPlusForward10_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefZeroBiasPlusForward5_datasetPPRefZeroBiasPlusForward10_selector.throw      = cms.bool(False)
streamPhysicsPPRefZeroBiasPlusForward5_datasetPPRefZeroBiasPlusForward10_selector.triggerConditions = cms.vstring(
    'HLT_PPRefUPC_SingleJet12_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet12_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet16_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet16_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet20_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet20_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet24_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet24_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet28_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet28_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet8_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet8_ZDC1nOR_v1',
    'HLT_PPRefUPC_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_ZDC1nOR_v1',
    'HLT_PPRefZeroBias_v6'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefZeroBiasPlusForward5_datasetPPRefZeroBiasPlusForward11_selector
streamPhysicsPPRefZeroBiasPlusForward5_datasetPPRefZeroBiasPlusForward11_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefZeroBiasPlusForward5_datasetPPRefZeroBiasPlusForward11_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefZeroBiasPlusForward5_datasetPPRefZeroBiasPlusForward11_selector.throw      = cms.bool(False)
streamPhysicsPPRefZeroBiasPlusForward5_datasetPPRefZeroBiasPlusForward11_selector.triggerConditions = cms.vstring(
    'HLT_PPRefUPC_SingleJet12_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet12_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet16_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet16_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet20_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet20_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet24_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet24_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet28_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet28_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet8_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet8_ZDC1nOR_v1',
    'HLT_PPRefUPC_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_ZDC1nOR_v1',
    'HLT_PPRefZeroBias_v6'
)


# stream PhysicsPPRefZeroBiasPlusForward6

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefZeroBiasPlusForward6_datasetPPRefZeroBiasPlusForward12_selector
streamPhysicsPPRefZeroBiasPlusForward6_datasetPPRefZeroBiasPlusForward12_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefZeroBiasPlusForward6_datasetPPRefZeroBiasPlusForward12_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefZeroBiasPlusForward6_datasetPPRefZeroBiasPlusForward12_selector.throw      = cms.bool(False)
streamPhysicsPPRefZeroBiasPlusForward6_datasetPPRefZeroBiasPlusForward12_selector.triggerConditions = cms.vstring(
    'HLT_PPRefUPC_SingleJet12_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet12_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet16_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet16_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet20_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet20_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet24_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet24_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet28_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet28_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet8_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet8_ZDC1nOR_v1',
    'HLT_PPRefUPC_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_ZDC1nOR_v1',
    'HLT_PPRefZeroBias_v6'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefZeroBiasPlusForward6_datasetPPRefZeroBiasPlusForward13_selector
streamPhysicsPPRefZeroBiasPlusForward6_datasetPPRefZeroBiasPlusForward13_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefZeroBiasPlusForward6_datasetPPRefZeroBiasPlusForward13_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefZeroBiasPlusForward6_datasetPPRefZeroBiasPlusForward13_selector.throw      = cms.bool(False)
streamPhysicsPPRefZeroBiasPlusForward6_datasetPPRefZeroBiasPlusForward13_selector.triggerConditions = cms.vstring(
    'HLT_PPRefUPC_SingleJet12_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet12_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet16_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet16_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet20_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet20_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet24_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet24_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet28_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet28_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet8_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet8_ZDC1nOR_v1',
    'HLT_PPRefUPC_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_ZDC1nOR_v1',
    'HLT_PPRefZeroBias_v6'
)


# stream PhysicsPPRefZeroBiasPlusForward7

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefZeroBiasPlusForward7_datasetPPRefZeroBiasPlusForward14_selector
streamPhysicsPPRefZeroBiasPlusForward7_datasetPPRefZeroBiasPlusForward14_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefZeroBiasPlusForward7_datasetPPRefZeroBiasPlusForward14_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefZeroBiasPlusForward7_datasetPPRefZeroBiasPlusForward14_selector.throw      = cms.bool(False)
streamPhysicsPPRefZeroBiasPlusForward7_datasetPPRefZeroBiasPlusForward14_selector.triggerConditions = cms.vstring(
    'HLT_PPRefUPC_SingleJet12_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet12_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet16_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet16_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet20_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet20_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet24_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet24_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet28_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet28_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet8_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet8_ZDC1nOR_v1',
    'HLT_PPRefUPC_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_ZDC1nOR_v1',
    'HLT_PPRefZeroBias_v6'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefZeroBiasPlusForward7_datasetPPRefZeroBiasPlusForward15_selector
streamPhysicsPPRefZeroBiasPlusForward7_datasetPPRefZeroBiasPlusForward15_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefZeroBiasPlusForward7_datasetPPRefZeroBiasPlusForward15_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefZeroBiasPlusForward7_datasetPPRefZeroBiasPlusForward15_selector.throw      = cms.bool(False)
streamPhysicsPPRefZeroBiasPlusForward7_datasetPPRefZeroBiasPlusForward15_selector.triggerConditions = cms.vstring(
    'HLT_PPRefUPC_SingleJet12_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet12_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet16_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet16_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet20_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet20_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet24_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet24_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet28_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet28_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet8_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet8_ZDC1nOR_v1',
    'HLT_PPRefUPC_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_ZDC1nOR_v1',
    'HLT_PPRefZeroBias_v6'
)


# stream PhysicsPPRefZeroBiasPlusForward8

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefZeroBiasPlusForward8_datasetPPRefZeroBiasPlusForward16_selector
streamPhysicsPPRefZeroBiasPlusForward8_datasetPPRefZeroBiasPlusForward16_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefZeroBiasPlusForward8_datasetPPRefZeroBiasPlusForward16_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefZeroBiasPlusForward8_datasetPPRefZeroBiasPlusForward16_selector.throw      = cms.bool(False)
streamPhysicsPPRefZeroBiasPlusForward8_datasetPPRefZeroBiasPlusForward16_selector.triggerConditions = cms.vstring(
    'HLT_PPRefUPC_SingleJet12_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet12_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet16_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet16_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet20_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet20_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet24_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet24_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet28_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet28_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet8_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet8_ZDC1nOR_v1',
    'HLT_PPRefUPC_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_ZDC1nOR_v1',
    'HLT_PPRefZeroBias_v6'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefZeroBiasPlusForward8_datasetPPRefZeroBiasPlusForward17_selector
streamPhysicsPPRefZeroBiasPlusForward8_datasetPPRefZeroBiasPlusForward17_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefZeroBiasPlusForward8_datasetPPRefZeroBiasPlusForward17_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefZeroBiasPlusForward8_datasetPPRefZeroBiasPlusForward17_selector.throw      = cms.bool(False)
streamPhysicsPPRefZeroBiasPlusForward8_datasetPPRefZeroBiasPlusForward17_selector.triggerConditions = cms.vstring(
    'HLT_PPRefUPC_SingleJet12_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet12_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet16_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet16_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet20_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet20_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet24_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet24_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet28_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet28_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet8_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet8_ZDC1nOR_v1',
    'HLT_PPRefUPC_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_ZDC1nOR_v1',
    'HLT_PPRefZeroBias_v6'
)


# stream PhysicsPPRefZeroBiasPlusForward9

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefZeroBiasPlusForward9_datasetPPRefZeroBiasPlusForward18_selector
streamPhysicsPPRefZeroBiasPlusForward9_datasetPPRefZeroBiasPlusForward18_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefZeroBiasPlusForward9_datasetPPRefZeroBiasPlusForward18_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefZeroBiasPlusForward9_datasetPPRefZeroBiasPlusForward18_selector.throw      = cms.bool(False)
streamPhysicsPPRefZeroBiasPlusForward9_datasetPPRefZeroBiasPlusForward18_selector.triggerConditions = cms.vstring(
    'HLT_PPRefUPC_SingleJet12_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet12_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet16_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet16_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet20_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet20_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet24_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet24_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet28_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet28_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet8_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet8_ZDC1nOR_v1',
    'HLT_PPRefUPC_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_ZDC1nOR_v1',
    'HLT_PPRefZeroBias_v6'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPPRefZeroBiasPlusForward9_datasetPPRefZeroBiasPlusForward19_selector
streamPhysicsPPRefZeroBiasPlusForward9_datasetPPRefZeroBiasPlusForward19_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPPRefZeroBiasPlusForward9_datasetPPRefZeroBiasPlusForward19_selector.l1tResults = cms.InputTag('')
streamPhysicsPPRefZeroBiasPlusForward9_datasetPPRefZeroBiasPlusForward19_selector.throw      = cms.bool(False)
streamPhysicsPPRefZeroBiasPlusForward9_datasetPPRefZeroBiasPlusForward19_selector.triggerConditions = cms.vstring(
    'HLT_PPRefUPC_SingleJet12_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet12_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet16_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet16_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet20_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet20_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet24_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet24_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet28_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet28_ZDC1nOR_v1',
    'HLT_PPRefUPC_SingleJet8_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_SingleJet8_ZDC1nOR_v1',
    'HLT_PPRefUPC_ZDC1nAsymXOR_v1',
    'HLT_PPRefUPC_ZDC1nOR_v1',
    'HLT_PPRefZeroBias_v6'
)


# stream PhysicsSpecialRandom0

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialRandom0_datasetSpecialRandom0_selector
streamPhysicsSpecialRandom0_datasetSpecialRandom0_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialRandom0_datasetSpecialRandom0_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialRandom0_datasetSpecialRandom0_selector.throw      = cms.bool(False)
streamPhysicsSpecialRandom0_datasetSpecialRandom0_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialRandom0_datasetSpecialRandom1_selector
streamPhysicsSpecialRandom0_datasetSpecialRandom1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialRandom0_datasetSpecialRandom1_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialRandom0_datasetSpecialRandom1_selector.throw      = cms.bool(False)
streamPhysicsSpecialRandom0_datasetSpecialRandom1_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')


# stream PhysicsSpecialRandom1

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialRandom1_datasetSpecialRandom2_selector
streamPhysicsSpecialRandom1_datasetSpecialRandom2_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialRandom1_datasetSpecialRandom2_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialRandom1_datasetSpecialRandom2_selector.throw      = cms.bool(False)
streamPhysicsSpecialRandom1_datasetSpecialRandom2_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialRandom1_datasetSpecialRandom3_selector
streamPhysicsSpecialRandom1_datasetSpecialRandom3_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialRandom1_datasetSpecialRandom3_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialRandom1_datasetSpecialRandom3_selector.throw      = cms.bool(False)
streamPhysicsSpecialRandom1_datasetSpecialRandom3_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')


# stream PhysicsSpecialRandom2

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialRandom2_datasetSpecialRandom4_selector
streamPhysicsSpecialRandom2_datasetSpecialRandom4_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialRandom2_datasetSpecialRandom4_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialRandom2_datasetSpecialRandom4_selector.throw      = cms.bool(False)
streamPhysicsSpecialRandom2_datasetSpecialRandom4_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialRandom2_datasetSpecialRandom5_selector
streamPhysicsSpecialRandom2_datasetSpecialRandom5_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialRandom2_datasetSpecialRandom5_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialRandom2_datasetSpecialRandom5_selector.throw      = cms.bool(False)
streamPhysicsSpecialRandom2_datasetSpecialRandom5_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')


# stream PhysicsSpecialRandom3

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialRandom3_datasetSpecialRandom6_selector
streamPhysicsSpecialRandom3_datasetSpecialRandom6_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialRandom3_datasetSpecialRandom6_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialRandom3_datasetSpecialRandom6_selector.throw      = cms.bool(False)
streamPhysicsSpecialRandom3_datasetSpecialRandom6_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialRandom3_datasetSpecialRandom7_selector
streamPhysicsSpecialRandom3_datasetSpecialRandom7_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialRandom3_datasetSpecialRandom7_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialRandom3_datasetSpecialRandom7_selector.throw      = cms.bool(False)
streamPhysicsSpecialRandom3_datasetSpecialRandom7_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')


# stream PhysicsSpecialRandom4

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialRandom4_datasetSpecialRandom8_selector
streamPhysicsSpecialRandom4_datasetSpecialRandom8_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialRandom4_datasetSpecialRandom8_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialRandom4_datasetSpecialRandom8_selector.throw      = cms.bool(False)
streamPhysicsSpecialRandom4_datasetSpecialRandom8_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialRandom4_datasetSpecialRandom9_selector
streamPhysicsSpecialRandom4_datasetSpecialRandom9_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialRandom4_datasetSpecialRandom9_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialRandom4_datasetSpecialRandom9_selector.throw      = cms.bool(False)
streamPhysicsSpecialRandom4_datasetSpecialRandom9_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')


# stream PhysicsSpecialRandom5

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialRandom5_datasetSpecialRandom10_selector
streamPhysicsSpecialRandom5_datasetSpecialRandom10_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialRandom5_datasetSpecialRandom10_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialRandom5_datasetSpecialRandom10_selector.throw      = cms.bool(False)
streamPhysicsSpecialRandom5_datasetSpecialRandom10_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialRandom5_datasetSpecialRandom11_selector
streamPhysicsSpecialRandom5_datasetSpecialRandom11_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialRandom5_datasetSpecialRandom11_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialRandom5_datasetSpecialRandom11_selector.throw      = cms.bool(False)
streamPhysicsSpecialRandom5_datasetSpecialRandom11_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')


# stream PhysicsSpecialRandom6

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialRandom6_datasetSpecialRandom12_selector
streamPhysicsSpecialRandom6_datasetSpecialRandom12_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialRandom6_datasetSpecialRandom12_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialRandom6_datasetSpecialRandom12_selector.throw      = cms.bool(False)
streamPhysicsSpecialRandom6_datasetSpecialRandom12_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialRandom6_datasetSpecialRandom13_selector
streamPhysicsSpecialRandom6_datasetSpecialRandom13_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialRandom6_datasetSpecialRandom13_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialRandom6_datasetSpecialRandom13_selector.throw      = cms.bool(False)
streamPhysicsSpecialRandom6_datasetSpecialRandom13_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')


# stream PhysicsSpecialRandom7

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialRandom7_datasetSpecialRandom14_selector
streamPhysicsSpecialRandom7_datasetSpecialRandom14_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialRandom7_datasetSpecialRandom14_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialRandom7_datasetSpecialRandom14_selector.throw      = cms.bool(False)
streamPhysicsSpecialRandom7_datasetSpecialRandom14_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialRandom7_datasetSpecialRandom15_selector
streamPhysicsSpecialRandom7_datasetSpecialRandom15_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialRandom7_datasetSpecialRandom15_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialRandom7_datasetSpecialRandom15_selector.throw      = cms.bool(False)
streamPhysicsSpecialRandom7_datasetSpecialRandom15_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')


# stream PhysicsSpecialRandom8

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialRandom8_datasetSpecialRandom16_selector
streamPhysicsSpecialRandom8_datasetSpecialRandom16_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialRandom8_datasetSpecialRandom16_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialRandom8_datasetSpecialRandom16_selector.throw      = cms.bool(False)
streamPhysicsSpecialRandom8_datasetSpecialRandom16_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialRandom8_datasetSpecialRandom17_selector
streamPhysicsSpecialRandom8_datasetSpecialRandom17_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialRandom8_datasetSpecialRandom17_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialRandom8_datasetSpecialRandom17_selector.throw      = cms.bool(False)
streamPhysicsSpecialRandom8_datasetSpecialRandom17_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')


# stream PhysicsSpecialRandom9

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialRandom9_datasetSpecialRandom18_selector
streamPhysicsSpecialRandom9_datasetSpecialRandom18_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialRandom9_datasetSpecialRandom18_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialRandom9_datasetSpecialRandom18_selector.throw      = cms.bool(False)
streamPhysicsSpecialRandom9_datasetSpecialRandom18_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialRandom9_datasetSpecialRandom19_selector
streamPhysicsSpecialRandom9_datasetSpecialRandom19_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialRandom9_datasetSpecialRandom19_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialRandom9_datasetSpecialRandom19_selector.throw      = cms.bool(False)
streamPhysicsSpecialRandom9_datasetSpecialRandom19_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')

