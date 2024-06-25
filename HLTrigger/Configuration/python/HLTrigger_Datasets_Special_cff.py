# /dev/CMSSW_14_0_0/Special

import FWCore.ParameterSet.Config as cms


# stream PhysicsCommissioning

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetCommissioning_selector
streamPhysicsCommissioning_datasetCommissioning_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetCommissioning_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetCommissioning_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetCommissioning_selector.triggerConditions = cms.vstring(
    'HLT_IsoTrackHB_v12',
    'HLT_IsoTrackHE_v12',
    'HLT_L1BptxXOR_v2',
    'HLT_L1SingleMuCosmics_EMTF_v2'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetCosmics_selector
streamPhysicsCommissioning_datasetCosmics_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetCosmics_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetCosmics_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetCosmics_selector.triggerConditions = cms.vstring(
    'HLT_L1SingleMu3_v3',
    'HLT_L1SingleMu5_v3',
    'HLT_L1SingleMu7_v3',
    'HLT_L1SingleMuCosmics_v6',
    'HLT_L1SingleMuOpen_DT_v4',
    'HLT_L1SingleMuOpen_v4'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetHLTPhysics_selector
streamPhysicsCommissioning_datasetHLTPhysics_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetHLTPhysics_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetHLTPhysics_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetHLTPhysics_selector.triggerConditions = cms.vstring('HLT_Physics_v12')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetHcalNZS_selector
streamPhysicsCommissioning_datasetHcalNZS_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetHcalNZS_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetHcalNZS_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetHcalNZS_selector.triggerConditions = cms.vstring(
    'HLT_HcalNZS_v19',
    'HLT_HcalPhiSym_v21'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetMinimumBias_selector
streamPhysicsCommissioning_datasetMinimumBias_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetMinimumBias_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetMinimumBias_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetMinimumBias_selector.triggerConditions = cms.vstring(
    'HLT_BptxOR_v4',
    'HLT_L1ETM120_v2',
    'HLT_L1ETM150_v2',
    'HLT_L1EXT_HCAL_LaserMon1_v3',
    'HLT_L1EXT_HCAL_LaserMon4_v3',
    'HLT_L1HTT120er_v2',
    'HLT_L1HTT160er_v2',
    'HLT_L1HTT200er_v2',
    'HLT_L1HTT255er_v2',
    'HLT_L1HTT280er_v2',
    'HLT_L1HTT320er_v2',
    'HLT_L1HTT360er_v2',
    'HLT_L1HTT400er_v2',
    'HLT_L1HTT450er_v2',
    'HLT_L1SingleEG10er2p5_v2',
    'HLT_L1SingleEG15er2p5_v2',
    'HLT_L1SingleEG26er2p5_v2',
    'HLT_L1SingleEG28er1p5_v2',
    'HLT_L1SingleEG28er2p1_v2',
    'HLT_L1SingleEG28er2p5_v2',
    'HLT_L1SingleEG34er2p5_v2',
    'HLT_L1SingleEG36er2p5_v2',
    'HLT_L1SingleEG38er2p5_v2',
    'HLT_L1SingleEG40er2p5_v2',
    'HLT_L1SingleEG42er2p5_v2',
    'HLT_L1SingleEG45er2p5_v2',
    'HLT_L1SingleEG50_v2',
    'HLT_L1SingleEG8er2p5_v2',
    'HLT_L1SingleJet10erHE_v3',
    'HLT_L1SingleJet120_v2',
    'HLT_L1SingleJet12erHE_v3',
    'HLT_L1SingleJet180_v2',
    'HLT_L1SingleJet200_v3',
    'HLT_L1SingleJet35_v3',
    'HLT_L1SingleJet60_v2',
    'HLT_L1SingleJet8erHE_v3',
    'HLT_L1SingleJet90_v2'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetMuonShower_selector
streamPhysicsCommissioning_datasetMuonShower_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetMuonShower_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetMuonShower_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetMuonShower_selector.triggerConditions = cms.vstring('HLT_CscCluster_Cosmic_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetNoBPTX_selector
streamPhysicsCommissioning_datasetNoBPTX_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetNoBPTX_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetNoBPTX_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetNoBPTX_selector.triggerConditions = cms.vstring(
    'HLT_CDC_L2cosmic_10_er1p0_v8',
    'HLT_CDC_L2cosmic_5p5_er1p0_v8',
    'HLT_L2Mu10_NoVertex_NoBPTX3BX_v12',
    'HLT_L2Mu10_NoVertex_NoBPTX_v13',
    'HLT_L2Mu40_NoVertex_3Sta_NoBPTX3BX_v12',
    'HLT_L2Mu45_NoVertex_3Sta_NoBPTX3BX_v11'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetZeroBias_selector
streamPhysicsCommissioning_datasetZeroBias_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetZeroBias_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetZeroBias_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetZeroBias_selector.triggerConditions = cms.vstring(
    'HLT_Random_v3',
    'HLT_ZeroBias_Alignment_v6',
    'HLT_ZeroBias_FirstBXAfterTrain_v8',
    'HLT_ZeroBias_FirstCollisionAfterAbortGap_v10',
    'HLT_ZeroBias_FirstCollisionInTrain_v9',
    'HLT_ZeroBias_IsolatedBunches_v10',
    'HLT_ZeroBias_LastCollisionInTrain_v8',
    'HLT_ZeroBias_v11'
)


# stream PhysicsSpecialHLTPhysics0

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics0_datasetSpecialHLTPhysics0_selector
streamPhysicsSpecialHLTPhysics0_datasetSpecialHLTPhysics0_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics0_datasetSpecialHLTPhysics0_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics0_datasetSpecialHLTPhysics0_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics0_datasetSpecialHLTPhysics0_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v5')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics0_datasetSpecialHLTPhysics1_selector
streamPhysicsSpecialHLTPhysics0_datasetSpecialHLTPhysics1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics0_datasetSpecialHLTPhysics1_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics0_datasetSpecialHLTPhysics1_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics0_datasetSpecialHLTPhysics1_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v5')


# stream PhysicsSpecialHLTPhysics1

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics1_datasetSpecialHLTPhysics2_selector
streamPhysicsSpecialHLTPhysics1_datasetSpecialHLTPhysics2_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics1_datasetSpecialHLTPhysics2_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics1_datasetSpecialHLTPhysics2_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics1_datasetSpecialHLTPhysics2_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v5')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics1_datasetSpecialHLTPhysics3_selector
streamPhysicsSpecialHLTPhysics1_datasetSpecialHLTPhysics3_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics1_datasetSpecialHLTPhysics3_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics1_datasetSpecialHLTPhysics3_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics1_datasetSpecialHLTPhysics3_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v5')


# stream PhysicsSpecialHLTPhysics10

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics10_datasetSpecialHLTPhysics20_selector
streamPhysicsSpecialHLTPhysics10_datasetSpecialHLTPhysics20_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics10_datasetSpecialHLTPhysics20_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics10_datasetSpecialHLTPhysics20_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics10_datasetSpecialHLTPhysics20_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v5')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics10_datasetSpecialHLTPhysics21_selector
streamPhysicsSpecialHLTPhysics10_datasetSpecialHLTPhysics21_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics10_datasetSpecialHLTPhysics21_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics10_datasetSpecialHLTPhysics21_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics10_datasetSpecialHLTPhysics21_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v5')


# stream PhysicsSpecialHLTPhysics11

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics11_datasetSpecialHLTPhysics22_selector
streamPhysicsSpecialHLTPhysics11_datasetSpecialHLTPhysics22_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics11_datasetSpecialHLTPhysics22_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics11_datasetSpecialHLTPhysics22_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics11_datasetSpecialHLTPhysics22_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v5')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics11_datasetSpecialHLTPhysics23_selector
streamPhysicsSpecialHLTPhysics11_datasetSpecialHLTPhysics23_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics11_datasetSpecialHLTPhysics23_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics11_datasetSpecialHLTPhysics23_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics11_datasetSpecialHLTPhysics23_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v5')


# stream PhysicsSpecialHLTPhysics12

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics12_datasetSpecialHLTPhysics24_selector
streamPhysicsSpecialHLTPhysics12_datasetSpecialHLTPhysics24_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics12_datasetSpecialHLTPhysics24_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics12_datasetSpecialHLTPhysics24_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics12_datasetSpecialHLTPhysics24_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v5')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics12_datasetSpecialHLTPhysics25_selector
streamPhysicsSpecialHLTPhysics12_datasetSpecialHLTPhysics25_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics12_datasetSpecialHLTPhysics25_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics12_datasetSpecialHLTPhysics25_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics12_datasetSpecialHLTPhysics25_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v5')


# stream PhysicsSpecialHLTPhysics13

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics13_datasetSpecialHLTPhysics26_selector
streamPhysicsSpecialHLTPhysics13_datasetSpecialHLTPhysics26_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics13_datasetSpecialHLTPhysics26_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics13_datasetSpecialHLTPhysics26_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics13_datasetSpecialHLTPhysics26_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v5')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics13_datasetSpecialHLTPhysics27_selector
streamPhysicsSpecialHLTPhysics13_datasetSpecialHLTPhysics27_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics13_datasetSpecialHLTPhysics27_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics13_datasetSpecialHLTPhysics27_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics13_datasetSpecialHLTPhysics27_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v5')


# stream PhysicsSpecialHLTPhysics14

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics14_datasetSpecialHLTPhysics28_selector
streamPhysicsSpecialHLTPhysics14_datasetSpecialHLTPhysics28_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics14_datasetSpecialHLTPhysics28_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics14_datasetSpecialHLTPhysics28_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics14_datasetSpecialHLTPhysics28_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v5')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics14_datasetSpecialHLTPhysics29_selector
streamPhysicsSpecialHLTPhysics14_datasetSpecialHLTPhysics29_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics14_datasetSpecialHLTPhysics29_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics14_datasetSpecialHLTPhysics29_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics14_datasetSpecialHLTPhysics29_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v5')


# stream PhysicsSpecialHLTPhysics15

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics15_datasetSpecialHLTPhysics30_selector
streamPhysicsSpecialHLTPhysics15_datasetSpecialHLTPhysics30_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics15_datasetSpecialHLTPhysics30_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics15_datasetSpecialHLTPhysics30_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics15_datasetSpecialHLTPhysics30_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v5')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics15_datasetSpecialHLTPhysics31_selector
streamPhysicsSpecialHLTPhysics15_datasetSpecialHLTPhysics31_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics15_datasetSpecialHLTPhysics31_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics15_datasetSpecialHLTPhysics31_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics15_datasetSpecialHLTPhysics31_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v5')


# stream PhysicsSpecialHLTPhysics2

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics2_datasetSpecialHLTPhysics4_selector
streamPhysicsSpecialHLTPhysics2_datasetSpecialHLTPhysics4_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics2_datasetSpecialHLTPhysics4_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics2_datasetSpecialHLTPhysics4_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics2_datasetSpecialHLTPhysics4_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v5')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics2_datasetSpecialHLTPhysics5_selector
streamPhysicsSpecialHLTPhysics2_datasetSpecialHLTPhysics5_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics2_datasetSpecialHLTPhysics5_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics2_datasetSpecialHLTPhysics5_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics2_datasetSpecialHLTPhysics5_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v5')


# stream PhysicsSpecialHLTPhysics3

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics3_datasetSpecialHLTPhysics6_selector
streamPhysicsSpecialHLTPhysics3_datasetSpecialHLTPhysics6_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics3_datasetSpecialHLTPhysics6_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics3_datasetSpecialHLTPhysics6_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics3_datasetSpecialHLTPhysics6_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v5')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics3_datasetSpecialHLTPhysics7_selector
streamPhysicsSpecialHLTPhysics3_datasetSpecialHLTPhysics7_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics3_datasetSpecialHLTPhysics7_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics3_datasetSpecialHLTPhysics7_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics3_datasetSpecialHLTPhysics7_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v5')


# stream PhysicsSpecialHLTPhysics4

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics4_datasetSpecialHLTPhysics8_selector
streamPhysicsSpecialHLTPhysics4_datasetSpecialHLTPhysics8_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics4_datasetSpecialHLTPhysics8_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics4_datasetSpecialHLTPhysics8_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics4_datasetSpecialHLTPhysics8_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v5')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics4_datasetSpecialHLTPhysics9_selector
streamPhysicsSpecialHLTPhysics4_datasetSpecialHLTPhysics9_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics4_datasetSpecialHLTPhysics9_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics4_datasetSpecialHLTPhysics9_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics4_datasetSpecialHLTPhysics9_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v5')


# stream PhysicsSpecialHLTPhysics5

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics5_datasetSpecialHLTPhysics10_selector
streamPhysicsSpecialHLTPhysics5_datasetSpecialHLTPhysics10_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics5_datasetSpecialHLTPhysics10_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics5_datasetSpecialHLTPhysics10_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics5_datasetSpecialHLTPhysics10_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v5')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics5_datasetSpecialHLTPhysics11_selector
streamPhysicsSpecialHLTPhysics5_datasetSpecialHLTPhysics11_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics5_datasetSpecialHLTPhysics11_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics5_datasetSpecialHLTPhysics11_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics5_datasetSpecialHLTPhysics11_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v5')


# stream PhysicsSpecialHLTPhysics6

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics6_datasetSpecialHLTPhysics12_selector
streamPhysicsSpecialHLTPhysics6_datasetSpecialHLTPhysics12_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics6_datasetSpecialHLTPhysics12_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics6_datasetSpecialHLTPhysics12_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics6_datasetSpecialHLTPhysics12_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v5')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics6_datasetSpecialHLTPhysics13_selector
streamPhysicsSpecialHLTPhysics6_datasetSpecialHLTPhysics13_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics6_datasetSpecialHLTPhysics13_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics6_datasetSpecialHLTPhysics13_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics6_datasetSpecialHLTPhysics13_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v5')


# stream PhysicsSpecialHLTPhysics7

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics7_datasetSpecialHLTPhysics14_selector
streamPhysicsSpecialHLTPhysics7_datasetSpecialHLTPhysics14_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics7_datasetSpecialHLTPhysics14_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics7_datasetSpecialHLTPhysics14_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics7_datasetSpecialHLTPhysics14_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v5')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics7_datasetSpecialHLTPhysics15_selector
streamPhysicsSpecialHLTPhysics7_datasetSpecialHLTPhysics15_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics7_datasetSpecialHLTPhysics15_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics7_datasetSpecialHLTPhysics15_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics7_datasetSpecialHLTPhysics15_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v5')


# stream PhysicsSpecialHLTPhysics8

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics8_datasetSpecialHLTPhysics16_selector
streamPhysicsSpecialHLTPhysics8_datasetSpecialHLTPhysics16_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics8_datasetSpecialHLTPhysics16_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics8_datasetSpecialHLTPhysics16_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics8_datasetSpecialHLTPhysics16_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v5')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics8_datasetSpecialHLTPhysics17_selector
streamPhysicsSpecialHLTPhysics8_datasetSpecialHLTPhysics17_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics8_datasetSpecialHLTPhysics17_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics8_datasetSpecialHLTPhysics17_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics8_datasetSpecialHLTPhysics17_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v5')


# stream PhysicsSpecialHLTPhysics9

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics9_datasetSpecialHLTPhysics18_selector
streamPhysicsSpecialHLTPhysics9_datasetSpecialHLTPhysics18_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics9_datasetSpecialHLTPhysics18_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics9_datasetSpecialHLTPhysics18_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics9_datasetSpecialHLTPhysics18_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v5')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics9_datasetSpecialHLTPhysics19_selector
streamPhysicsSpecialHLTPhysics9_datasetSpecialHLTPhysics19_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics9_datasetSpecialHLTPhysics19_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics9_datasetSpecialHLTPhysics19_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics9_datasetSpecialHLTPhysics19_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v5')


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


# stream PhysicsSpecialRandom10

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialRandom10_datasetSpecialRandom20_selector
streamPhysicsSpecialRandom10_datasetSpecialRandom20_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialRandom10_datasetSpecialRandom20_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialRandom10_datasetSpecialRandom20_selector.throw      = cms.bool(False)
streamPhysicsSpecialRandom10_datasetSpecialRandom20_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialRandom10_datasetSpecialRandom21_selector
streamPhysicsSpecialRandom10_datasetSpecialRandom21_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialRandom10_datasetSpecialRandom21_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialRandom10_datasetSpecialRandom21_selector.throw      = cms.bool(False)
streamPhysicsSpecialRandom10_datasetSpecialRandom21_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')


# stream PhysicsSpecialRandom11

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialRandom11_datasetSpecialRandom22_selector
streamPhysicsSpecialRandom11_datasetSpecialRandom22_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialRandom11_datasetSpecialRandom22_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialRandom11_datasetSpecialRandom22_selector.throw      = cms.bool(False)
streamPhysicsSpecialRandom11_datasetSpecialRandom22_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialRandom11_datasetSpecialRandom23_selector
streamPhysicsSpecialRandom11_datasetSpecialRandom23_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialRandom11_datasetSpecialRandom23_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialRandom11_datasetSpecialRandom23_selector.throw      = cms.bool(False)
streamPhysicsSpecialRandom11_datasetSpecialRandom23_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')


# stream PhysicsSpecialRandom12

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialRandom12_datasetSpecialRandom24_selector
streamPhysicsSpecialRandom12_datasetSpecialRandom24_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialRandom12_datasetSpecialRandom24_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialRandom12_datasetSpecialRandom24_selector.throw      = cms.bool(False)
streamPhysicsSpecialRandom12_datasetSpecialRandom24_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialRandom12_datasetSpecialRandom25_selector
streamPhysicsSpecialRandom12_datasetSpecialRandom25_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialRandom12_datasetSpecialRandom25_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialRandom12_datasetSpecialRandom25_selector.throw      = cms.bool(False)
streamPhysicsSpecialRandom12_datasetSpecialRandom25_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')


# stream PhysicsSpecialRandom13

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialRandom13_datasetSpecialRandom26_selector
streamPhysicsSpecialRandom13_datasetSpecialRandom26_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialRandom13_datasetSpecialRandom26_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialRandom13_datasetSpecialRandom26_selector.throw      = cms.bool(False)
streamPhysicsSpecialRandom13_datasetSpecialRandom26_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialRandom13_datasetSpecialRandom27_selector
streamPhysicsSpecialRandom13_datasetSpecialRandom27_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialRandom13_datasetSpecialRandom27_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialRandom13_datasetSpecialRandom27_selector.throw      = cms.bool(False)
streamPhysicsSpecialRandom13_datasetSpecialRandom27_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')


# stream PhysicsSpecialRandom14

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialRandom14_datasetSpecialRandom28_selector
streamPhysicsSpecialRandom14_datasetSpecialRandom28_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialRandom14_datasetSpecialRandom28_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialRandom14_datasetSpecialRandom28_selector.throw      = cms.bool(False)
streamPhysicsSpecialRandom14_datasetSpecialRandom28_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialRandom14_datasetSpecialRandom29_selector
streamPhysicsSpecialRandom14_datasetSpecialRandom29_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialRandom14_datasetSpecialRandom29_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialRandom14_datasetSpecialRandom29_selector.throw      = cms.bool(False)
streamPhysicsSpecialRandom14_datasetSpecialRandom29_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')


# stream PhysicsSpecialRandom15

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialRandom15_datasetSpecialRandom30_selector
streamPhysicsSpecialRandom15_datasetSpecialRandom30_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialRandom15_datasetSpecialRandom30_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialRandom15_datasetSpecialRandom30_selector.throw      = cms.bool(False)
streamPhysicsSpecialRandom15_datasetSpecialRandom30_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialRandom15_datasetSpecialRandom31_selector
streamPhysicsSpecialRandom15_datasetSpecialRandom31_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialRandom15_datasetSpecialRandom31_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialRandom15_datasetSpecialRandom31_selector.throw      = cms.bool(False)
streamPhysicsSpecialRandom15_datasetSpecialRandom31_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')


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


# stream PhysicsSpecialZeroBias0

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias0_datasetSpecialZeroBias0_selector
streamPhysicsSpecialZeroBias0_datasetSpecialZeroBias0_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias0_datasetSpecialZeroBias0_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias0_datasetSpecialZeroBias0_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias0_datasetSpecialZeroBias0_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v4',
    'HLT_ZeroBias_Gated_v2',
    'HLT_ZeroBias_HighRate_v2'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias0_datasetSpecialZeroBias1_selector
streamPhysicsSpecialZeroBias0_datasetSpecialZeroBias1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias0_datasetSpecialZeroBias1_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias0_datasetSpecialZeroBias1_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias0_datasetSpecialZeroBias1_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v4',
    'HLT_ZeroBias_Gated_v2',
    'HLT_ZeroBias_HighRate_v2'
)


# stream PhysicsSpecialZeroBias1

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias1_datasetSpecialZeroBias2_selector
streamPhysicsSpecialZeroBias1_datasetSpecialZeroBias2_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias1_datasetSpecialZeroBias2_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias1_datasetSpecialZeroBias2_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias1_datasetSpecialZeroBias2_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v4',
    'HLT_ZeroBias_Gated_v2',
    'HLT_ZeroBias_HighRate_v2'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias1_datasetSpecialZeroBias3_selector
streamPhysicsSpecialZeroBias1_datasetSpecialZeroBias3_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias1_datasetSpecialZeroBias3_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias1_datasetSpecialZeroBias3_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias1_datasetSpecialZeroBias3_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v4',
    'HLT_ZeroBias_Gated_v2',
    'HLT_ZeroBias_HighRate_v2'
)


# stream PhysicsSpecialZeroBias10

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias10_datasetSpecialZeroBias20_selector
streamPhysicsSpecialZeroBias10_datasetSpecialZeroBias20_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias10_datasetSpecialZeroBias20_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias10_datasetSpecialZeroBias20_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias10_datasetSpecialZeroBias20_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v4',
    'HLT_ZeroBias_Gated_v2',
    'HLT_ZeroBias_HighRate_v2'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias10_datasetSpecialZeroBias21_selector
streamPhysicsSpecialZeroBias10_datasetSpecialZeroBias21_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias10_datasetSpecialZeroBias21_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias10_datasetSpecialZeroBias21_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias10_datasetSpecialZeroBias21_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v4',
    'HLT_ZeroBias_Gated_v2',
    'HLT_ZeroBias_HighRate_v2'
)


# stream PhysicsSpecialZeroBias11

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias11_datasetSpecialZeroBias22_selector
streamPhysicsSpecialZeroBias11_datasetSpecialZeroBias22_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias11_datasetSpecialZeroBias22_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias11_datasetSpecialZeroBias22_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias11_datasetSpecialZeroBias22_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v4',
    'HLT_ZeroBias_Gated_v2',
    'HLT_ZeroBias_HighRate_v2'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias11_datasetSpecialZeroBias23_selector
streamPhysicsSpecialZeroBias11_datasetSpecialZeroBias23_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias11_datasetSpecialZeroBias23_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias11_datasetSpecialZeroBias23_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias11_datasetSpecialZeroBias23_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v4',
    'HLT_ZeroBias_Gated_v2',
    'HLT_ZeroBias_HighRate_v2'
)


# stream PhysicsSpecialZeroBias12

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias12_datasetSpecialZeroBias24_selector
streamPhysicsSpecialZeroBias12_datasetSpecialZeroBias24_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias12_datasetSpecialZeroBias24_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias12_datasetSpecialZeroBias24_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias12_datasetSpecialZeroBias24_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v4',
    'HLT_ZeroBias_Gated_v2',
    'HLT_ZeroBias_HighRate_v2'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias12_datasetSpecialZeroBias25_selector
streamPhysicsSpecialZeroBias12_datasetSpecialZeroBias25_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias12_datasetSpecialZeroBias25_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias12_datasetSpecialZeroBias25_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias12_datasetSpecialZeroBias25_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v4',
    'HLT_ZeroBias_Gated_v2',
    'HLT_ZeroBias_HighRate_v2'
)


# stream PhysicsSpecialZeroBias13

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias13_datasetSpecialZeroBias26_selector
streamPhysicsSpecialZeroBias13_datasetSpecialZeroBias26_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias13_datasetSpecialZeroBias26_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias13_datasetSpecialZeroBias26_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias13_datasetSpecialZeroBias26_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v4',
    'HLT_ZeroBias_Gated_v2',
    'HLT_ZeroBias_HighRate_v2'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias13_datasetSpecialZeroBias27_selector
streamPhysicsSpecialZeroBias13_datasetSpecialZeroBias27_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias13_datasetSpecialZeroBias27_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias13_datasetSpecialZeroBias27_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias13_datasetSpecialZeroBias27_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v4',
    'HLT_ZeroBias_Gated_v2',
    'HLT_ZeroBias_HighRate_v2'
)


# stream PhysicsSpecialZeroBias14

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias14_datasetSpecialZeroBias28_selector
streamPhysicsSpecialZeroBias14_datasetSpecialZeroBias28_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias14_datasetSpecialZeroBias28_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias14_datasetSpecialZeroBias28_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias14_datasetSpecialZeroBias28_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v4',
    'HLT_ZeroBias_Gated_v2',
    'HLT_ZeroBias_HighRate_v2'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias14_datasetSpecialZeroBias29_selector
streamPhysicsSpecialZeroBias14_datasetSpecialZeroBias29_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias14_datasetSpecialZeroBias29_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias14_datasetSpecialZeroBias29_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias14_datasetSpecialZeroBias29_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v4',
    'HLT_ZeroBias_Gated_v2',
    'HLT_ZeroBias_HighRate_v2'
)


# stream PhysicsSpecialZeroBias15

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias15_datasetSpecialZeroBias30_selector
streamPhysicsSpecialZeroBias15_datasetSpecialZeroBias30_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias15_datasetSpecialZeroBias30_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias15_datasetSpecialZeroBias30_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias15_datasetSpecialZeroBias30_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v4',
    'HLT_ZeroBias_Gated_v2',
    'HLT_ZeroBias_HighRate_v2'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias15_datasetSpecialZeroBias31_selector
streamPhysicsSpecialZeroBias15_datasetSpecialZeroBias31_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias15_datasetSpecialZeroBias31_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias15_datasetSpecialZeroBias31_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias15_datasetSpecialZeroBias31_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v4',
    'HLT_ZeroBias_Gated_v2',
    'HLT_ZeroBias_HighRate_v2'
)


# stream PhysicsSpecialZeroBias2

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias2_datasetSpecialZeroBias4_selector
streamPhysicsSpecialZeroBias2_datasetSpecialZeroBias4_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias2_datasetSpecialZeroBias4_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias2_datasetSpecialZeroBias4_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias2_datasetSpecialZeroBias4_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v4',
    'HLT_ZeroBias_Gated_v2',
    'HLT_ZeroBias_HighRate_v2'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias2_datasetSpecialZeroBias5_selector
streamPhysicsSpecialZeroBias2_datasetSpecialZeroBias5_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias2_datasetSpecialZeroBias5_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias2_datasetSpecialZeroBias5_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias2_datasetSpecialZeroBias5_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v4',
    'HLT_ZeroBias_Gated_v2',
    'HLT_ZeroBias_HighRate_v2'
)


# stream PhysicsSpecialZeroBias3

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias3_datasetSpecialZeroBias6_selector
streamPhysicsSpecialZeroBias3_datasetSpecialZeroBias6_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias3_datasetSpecialZeroBias6_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias3_datasetSpecialZeroBias6_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias3_datasetSpecialZeroBias6_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v4',
    'HLT_ZeroBias_Gated_v2',
    'HLT_ZeroBias_HighRate_v2'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias3_datasetSpecialZeroBias7_selector
streamPhysicsSpecialZeroBias3_datasetSpecialZeroBias7_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias3_datasetSpecialZeroBias7_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias3_datasetSpecialZeroBias7_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias3_datasetSpecialZeroBias7_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v4',
    'HLT_ZeroBias_Gated_v2',
    'HLT_ZeroBias_HighRate_v2'
)


# stream PhysicsSpecialZeroBias4

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias4_datasetSpecialZeroBias8_selector
streamPhysicsSpecialZeroBias4_datasetSpecialZeroBias8_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias4_datasetSpecialZeroBias8_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias4_datasetSpecialZeroBias8_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias4_datasetSpecialZeroBias8_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v4',
    'HLT_ZeroBias_Gated_v2',
    'HLT_ZeroBias_HighRate_v2'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias4_datasetSpecialZeroBias9_selector
streamPhysicsSpecialZeroBias4_datasetSpecialZeroBias9_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias4_datasetSpecialZeroBias9_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias4_datasetSpecialZeroBias9_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias4_datasetSpecialZeroBias9_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v4',
    'HLT_ZeroBias_Gated_v2',
    'HLT_ZeroBias_HighRate_v2'
)


# stream PhysicsSpecialZeroBias5

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias5_datasetSpecialZeroBias10_selector
streamPhysicsSpecialZeroBias5_datasetSpecialZeroBias10_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias5_datasetSpecialZeroBias10_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias5_datasetSpecialZeroBias10_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias5_datasetSpecialZeroBias10_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v4',
    'HLT_ZeroBias_Gated_v2',
    'HLT_ZeroBias_HighRate_v2'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias5_datasetSpecialZeroBias11_selector
streamPhysicsSpecialZeroBias5_datasetSpecialZeroBias11_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias5_datasetSpecialZeroBias11_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias5_datasetSpecialZeroBias11_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias5_datasetSpecialZeroBias11_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v4',
    'HLT_ZeroBias_Gated_v2',
    'HLT_ZeroBias_HighRate_v2'
)


# stream PhysicsSpecialZeroBias6

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias6_datasetSpecialZeroBias12_selector
streamPhysicsSpecialZeroBias6_datasetSpecialZeroBias12_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias6_datasetSpecialZeroBias12_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias6_datasetSpecialZeroBias12_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias6_datasetSpecialZeroBias12_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v4',
    'HLT_ZeroBias_Gated_v2',
    'HLT_ZeroBias_HighRate_v2'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias6_datasetSpecialZeroBias13_selector
streamPhysicsSpecialZeroBias6_datasetSpecialZeroBias13_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias6_datasetSpecialZeroBias13_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias6_datasetSpecialZeroBias13_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias6_datasetSpecialZeroBias13_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v4',
    'HLT_ZeroBias_Gated_v2',
    'HLT_ZeroBias_HighRate_v2'
)


# stream PhysicsSpecialZeroBias7

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias7_datasetSpecialZeroBias14_selector
streamPhysicsSpecialZeroBias7_datasetSpecialZeroBias14_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias7_datasetSpecialZeroBias14_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias7_datasetSpecialZeroBias14_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias7_datasetSpecialZeroBias14_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v4',
    'HLT_ZeroBias_Gated_v2',
    'HLT_ZeroBias_HighRate_v2'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias7_datasetSpecialZeroBias15_selector
streamPhysicsSpecialZeroBias7_datasetSpecialZeroBias15_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias7_datasetSpecialZeroBias15_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias7_datasetSpecialZeroBias15_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias7_datasetSpecialZeroBias15_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v4',
    'HLT_ZeroBias_Gated_v2',
    'HLT_ZeroBias_HighRate_v2'
)


# stream PhysicsSpecialZeroBias8

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias8_datasetSpecialZeroBias16_selector
streamPhysicsSpecialZeroBias8_datasetSpecialZeroBias16_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias8_datasetSpecialZeroBias16_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias8_datasetSpecialZeroBias16_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias8_datasetSpecialZeroBias16_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v4',
    'HLT_ZeroBias_Gated_v2',
    'HLT_ZeroBias_HighRate_v2'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias8_datasetSpecialZeroBias17_selector
streamPhysicsSpecialZeroBias8_datasetSpecialZeroBias17_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias8_datasetSpecialZeroBias17_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias8_datasetSpecialZeroBias17_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias8_datasetSpecialZeroBias17_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v4',
    'HLT_ZeroBias_Gated_v2',
    'HLT_ZeroBias_HighRate_v2'
)


# stream PhysicsSpecialZeroBias9

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias9_datasetSpecialZeroBias18_selector
streamPhysicsSpecialZeroBias9_datasetSpecialZeroBias18_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias9_datasetSpecialZeroBias18_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias9_datasetSpecialZeroBias18_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias9_datasetSpecialZeroBias18_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v4',
    'HLT_ZeroBias_Gated_v2',
    'HLT_ZeroBias_HighRate_v2'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias9_datasetSpecialZeroBias19_selector
streamPhysicsSpecialZeroBias9_datasetSpecialZeroBias19_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias9_datasetSpecialZeroBias19_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias9_datasetSpecialZeroBias19_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias9_datasetSpecialZeroBias19_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v4',
    'HLT_ZeroBias_Gated_v2',
    'HLT_ZeroBias_HighRate_v2'
)

