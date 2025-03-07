# /dev/CMSSW_15_0_0/Special

import FWCore.ParameterSet.Config as cms


# stream PhysicsCommissioning

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetCommissioning_selector
streamPhysicsCommissioning_datasetCommissioning_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetCommissioning_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetCommissioning_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetCommissioning_selector.triggerConditions = cms.vstring(
    'HLT_IsoTrackHB_v14',
    'HLT_IsoTrackHE_v14',
    'HLT_L1SingleMuCosmics_EMTF_v4'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetCosmics_selector
streamPhysicsCommissioning_datasetCosmics_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetCosmics_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetCosmics_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetCosmics_selector.triggerConditions = cms.vstring(
    'HLT_L1SingleMu3_v5',
    'HLT_L1SingleMu5_v5',
    'HLT_L1SingleMu7_v5',
    'HLT_L1SingleMuCosmics_v8',
    'HLT_L1SingleMuOpen_DT_v6',
    'HLT_L1SingleMuOpen_v6'
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

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetMinimumBias_selector
streamPhysicsCommissioning_datasetMinimumBias_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetMinimumBias_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetMinimumBias_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetMinimumBias_selector.triggerConditions = cms.vstring(
    'HLT_BptxOR_v6',
    'HLT_L1ETM120_v4',
    'HLT_L1ETM150_v4',
    'HLT_L1EXT_HCAL_LaserMon1_v5',
    'HLT_L1EXT_HCAL_LaserMon4_v5',
    'HLT_L1HTT120er_v4',
    'HLT_L1HTT160er_v4',
    'HLT_L1HTT200er_v4',
    'HLT_L1HTT255er_v4',
    'HLT_L1HTT280er_v4',
    'HLT_L1HTT320er_v4',
    'HLT_L1HTT360er_v4',
    'HLT_L1HTT400er_v4',
    'HLT_L1HTT450er_v4',
    'HLT_L1SingleEG10er2p5_v4',
    'HLT_L1SingleEG15er2p5_v4',
    'HLT_L1SingleEG26er2p5_v4',
    'HLT_L1SingleEG28er1p5_v4',
    'HLT_L1SingleEG28er2p1_v4',
    'HLT_L1SingleEG28er2p5_v4',
    'HLT_L1SingleEG34er2p5_v4',
    'HLT_L1SingleEG36er2p5_v4',
    'HLT_L1SingleEG38er2p5_v4',
    'HLT_L1SingleEG40er2p5_v4',
    'HLT_L1SingleEG42er2p5_v4',
    'HLT_L1SingleEG45er2p5_v4',
    'HLT_L1SingleEG50_v4',
    'HLT_L1SingleEG8er2p5_v4',
    'HLT_L1SingleJet10erHE_v5',
    'HLT_L1SingleJet120_v4',
    'HLT_L1SingleJet12erHE_v5',
    'HLT_L1SingleJet180_v4',
    'HLT_L1SingleJet200_v5',
    'HLT_L1SingleJet35_v5',
    'HLT_L1SingleJet60_v4',
    'HLT_L1SingleJet8erHE_v5',
    'HLT_L1SingleJet90_v4'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetMuonShower_selector
streamPhysicsCommissioning_datasetMuonShower_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetMuonShower_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetMuonShower_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetMuonShower_selector.triggerConditions = cms.vstring('HLT_CscCluster_Cosmic_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetNoBPTX_selector
streamPhysicsCommissioning_datasetNoBPTX_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetNoBPTX_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetNoBPTX_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetNoBPTX_selector.triggerConditions = cms.vstring(
    'HLT_CDC_L2cosmic_10_er1p0_v10',
    'HLT_CDC_L2cosmic_5p5_er1p0_v10',
    'HLT_L2Mu10_NoVertex_NoBPTX3BX_v14',
    'HLT_L2Mu10_NoVertex_NoBPTX_v15',
    'HLT_L2Mu40_NoVertex_3Sta_NoBPTX3BX_v14',
    'HLT_L2Mu45_NoVertex_3Sta_NoBPTX3BX_v13'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetZeroBias_selector
streamPhysicsCommissioning_datasetZeroBias_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetZeroBias_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetZeroBias_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetZeroBias_selector.triggerConditions = cms.vstring(
    'HLT_Random_v3',
    'HLT_ZeroBias_Alignment_v8',
    'HLT_ZeroBias_FirstBXAfterTrain_v10',
    'HLT_ZeroBias_FirstCollisionAfterAbortGap_v12',
    'HLT_ZeroBias_FirstCollisionInTrain_v11',
    'HLT_ZeroBias_IsolatedBunches_v12',
    'HLT_ZeroBias_LastCollisionInTrain_v10',
    'HLT_ZeroBias_v13'
)


# stream PhysicsSpecialHLTPhysics0

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics0_datasetSpecialHLTPhysics0_selector
streamPhysicsSpecialHLTPhysics0_datasetSpecialHLTPhysics0_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics0_datasetSpecialHLTPhysics0_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics0_datasetSpecialHLTPhysics0_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics0_datasetSpecialHLTPhysics0_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v7')


# stream PhysicsSpecialHLTPhysics1

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics1_datasetSpecialHLTPhysics1_selector
streamPhysicsSpecialHLTPhysics1_datasetSpecialHLTPhysics1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics1_datasetSpecialHLTPhysics1_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics1_datasetSpecialHLTPhysics1_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics1_datasetSpecialHLTPhysics1_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v7')


# stream PhysicsSpecialHLTPhysics10

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics10_datasetSpecialHLTPhysics10_selector
streamPhysicsSpecialHLTPhysics10_datasetSpecialHLTPhysics10_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics10_datasetSpecialHLTPhysics10_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics10_datasetSpecialHLTPhysics10_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics10_datasetSpecialHLTPhysics10_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v7')


# stream PhysicsSpecialHLTPhysics11

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics11_datasetSpecialHLTPhysics11_selector
streamPhysicsSpecialHLTPhysics11_datasetSpecialHLTPhysics11_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics11_datasetSpecialHLTPhysics11_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics11_datasetSpecialHLTPhysics11_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics11_datasetSpecialHLTPhysics11_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v7')


# stream PhysicsSpecialHLTPhysics12

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics12_datasetSpecialHLTPhysics12_selector
streamPhysicsSpecialHLTPhysics12_datasetSpecialHLTPhysics12_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics12_datasetSpecialHLTPhysics12_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics12_datasetSpecialHLTPhysics12_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics12_datasetSpecialHLTPhysics12_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v7')


# stream PhysicsSpecialHLTPhysics13

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics13_datasetSpecialHLTPhysics13_selector
streamPhysicsSpecialHLTPhysics13_datasetSpecialHLTPhysics13_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics13_datasetSpecialHLTPhysics13_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics13_datasetSpecialHLTPhysics13_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics13_datasetSpecialHLTPhysics13_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v7')


# stream PhysicsSpecialHLTPhysics14

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics14_datasetSpecialHLTPhysics14_selector
streamPhysicsSpecialHLTPhysics14_datasetSpecialHLTPhysics14_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics14_datasetSpecialHLTPhysics14_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics14_datasetSpecialHLTPhysics14_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics14_datasetSpecialHLTPhysics14_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v7')


# stream PhysicsSpecialHLTPhysics15

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics15_datasetSpecialHLTPhysics15_selector
streamPhysicsSpecialHLTPhysics15_datasetSpecialHLTPhysics15_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics15_datasetSpecialHLTPhysics15_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics15_datasetSpecialHLTPhysics15_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics15_datasetSpecialHLTPhysics15_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v7')


# stream PhysicsSpecialHLTPhysics16

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics16_datasetSpecialHLTPhysics16_selector
streamPhysicsSpecialHLTPhysics16_datasetSpecialHLTPhysics16_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics16_datasetSpecialHLTPhysics16_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics16_datasetSpecialHLTPhysics16_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics16_datasetSpecialHLTPhysics16_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v7')


# stream PhysicsSpecialHLTPhysics17

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics17_datasetSpecialHLTPhysics17_selector
streamPhysicsSpecialHLTPhysics17_datasetSpecialHLTPhysics17_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics17_datasetSpecialHLTPhysics17_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics17_datasetSpecialHLTPhysics17_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics17_datasetSpecialHLTPhysics17_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v7')


# stream PhysicsSpecialHLTPhysics18

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics18_datasetSpecialHLTPhysics18_selector
streamPhysicsSpecialHLTPhysics18_datasetSpecialHLTPhysics18_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics18_datasetSpecialHLTPhysics18_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics18_datasetSpecialHLTPhysics18_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics18_datasetSpecialHLTPhysics18_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v7')


# stream PhysicsSpecialHLTPhysics19

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics19_datasetSpecialHLTPhysics19_selector
streamPhysicsSpecialHLTPhysics19_datasetSpecialHLTPhysics19_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics19_datasetSpecialHLTPhysics19_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics19_datasetSpecialHLTPhysics19_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics19_datasetSpecialHLTPhysics19_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v7')


# stream PhysicsSpecialHLTPhysics2

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics2_datasetSpecialHLTPhysics2_selector
streamPhysicsSpecialHLTPhysics2_datasetSpecialHLTPhysics2_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics2_datasetSpecialHLTPhysics2_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics2_datasetSpecialHLTPhysics2_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics2_datasetSpecialHLTPhysics2_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v7')


# stream PhysicsSpecialHLTPhysics3

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics3_datasetSpecialHLTPhysics3_selector
streamPhysicsSpecialHLTPhysics3_datasetSpecialHLTPhysics3_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics3_datasetSpecialHLTPhysics3_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics3_datasetSpecialHLTPhysics3_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics3_datasetSpecialHLTPhysics3_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v7')


# stream PhysicsSpecialHLTPhysics4

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics4_datasetSpecialHLTPhysics4_selector
streamPhysicsSpecialHLTPhysics4_datasetSpecialHLTPhysics4_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics4_datasetSpecialHLTPhysics4_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics4_datasetSpecialHLTPhysics4_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics4_datasetSpecialHLTPhysics4_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v7')


# stream PhysicsSpecialHLTPhysics5

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics5_datasetSpecialHLTPhysics5_selector
streamPhysicsSpecialHLTPhysics5_datasetSpecialHLTPhysics5_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics5_datasetSpecialHLTPhysics5_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics5_datasetSpecialHLTPhysics5_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics5_datasetSpecialHLTPhysics5_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v7')


# stream PhysicsSpecialHLTPhysics6

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics6_datasetSpecialHLTPhysics6_selector
streamPhysicsSpecialHLTPhysics6_datasetSpecialHLTPhysics6_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics6_datasetSpecialHLTPhysics6_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics6_datasetSpecialHLTPhysics6_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics6_datasetSpecialHLTPhysics6_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v7')


# stream PhysicsSpecialHLTPhysics7

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics7_datasetSpecialHLTPhysics7_selector
streamPhysicsSpecialHLTPhysics7_datasetSpecialHLTPhysics7_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics7_datasetSpecialHLTPhysics7_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics7_datasetSpecialHLTPhysics7_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics7_datasetSpecialHLTPhysics7_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v7')


# stream PhysicsSpecialHLTPhysics8

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics8_datasetSpecialHLTPhysics8_selector
streamPhysicsSpecialHLTPhysics8_datasetSpecialHLTPhysics8_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics8_datasetSpecialHLTPhysics8_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics8_datasetSpecialHLTPhysics8_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics8_datasetSpecialHLTPhysics8_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v7')


# stream PhysicsSpecialHLTPhysics9

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialHLTPhysics9_datasetSpecialHLTPhysics9_selector
streamPhysicsSpecialHLTPhysics9_datasetSpecialHLTPhysics9_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialHLTPhysics9_datasetSpecialHLTPhysics9_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialHLTPhysics9_datasetSpecialHLTPhysics9_selector.throw      = cms.bool(False)
streamPhysicsSpecialHLTPhysics9_datasetSpecialHLTPhysics9_selector.triggerConditions = cms.vstring('HLT_SpecialHLTPhysics_v7')


# stream PhysicsSpecialMinimumBias0

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialMinimumBias0_datasetSpecialMinimumBias0_selector
streamPhysicsSpecialMinimumBias0_datasetSpecialMinimumBias0_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialMinimumBias0_datasetSpecialMinimumBias0_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialMinimumBias0_datasetSpecialMinimumBias0_selector.throw      = cms.bool(False)
streamPhysicsSpecialMinimumBias0_datasetSpecialMinimumBias0_selector.triggerConditions = cms.vstring(
    'HLT_L1MinimumBiasHF0ANDBptxAND_v1',
    'HLT_PixelClusters_WP2_HighRate_v1'
)


# stream PhysicsSpecialMinimumBias1

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialMinimumBias1_datasetSpecialMinimumBias1_selector
streamPhysicsSpecialMinimumBias1_datasetSpecialMinimumBias1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialMinimumBias1_datasetSpecialMinimumBias1_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialMinimumBias1_datasetSpecialMinimumBias1_selector.throw      = cms.bool(False)
streamPhysicsSpecialMinimumBias1_datasetSpecialMinimumBias1_selector.triggerConditions = cms.vstring(
    'HLT_L1MinimumBiasHF0ANDBptxAND_v1',
    'HLT_PixelClusters_WP2_HighRate_v1'
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


# stream PhysicsSpecialZeroBias0

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias0_datasetSpecialZeroBias0_selector
streamPhysicsSpecialZeroBias0_datasetSpecialZeroBias0_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias0_datasetSpecialZeroBias0_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias0_datasetSpecialZeroBias0_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias0_datasetSpecialZeroBias0_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias0_datasetSpecialZeroBias1_selector
streamPhysicsSpecialZeroBias0_datasetSpecialZeroBias1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias0_datasetSpecialZeroBias1_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias0_datasetSpecialZeroBias1_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias0_datasetSpecialZeroBias1_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4'
)


# stream PhysicsSpecialZeroBias1

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias1_datasetSpecialZeroBias2_selector
streamPhysicsSpecialZeroBias1_datasetSpecialZeroBias2_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias1_datasetSpecialZeroBias2_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias1_datasetSpecialZeroBias2_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias1_datasetSpecialZeroBias2_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias1_datasetSpecialZeroBias3_selector
streamPhysicsSpecialZeroBias1_datasetSpecialZeroBias3_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias1_datasetSpecialZeroBias3_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias1_datasetSpecialZeroBias3_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias1_datasetSpecialZeroBias3_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4'
)


# stream PhysicsSpecialZeroBias10

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias10_datasetSpecialZeroBias20_selector
streamPhysicsSpecialZeroBias10_datasetSpecialZeroBias20_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias10_datasetSpecialZeroBias20_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias10_datasetSpecialZeroBias20_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias10_datasetSpecialZeroBias20_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias10_datasetSpecialZeroBias21_selector
streamPhysicsSpecialZeroBias10_datasetSpecialZeroBias21_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias10_datasetSpecialZeroBias21_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias10_datasetSpecialZeroBias21_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias10_datasetSpecialZeroBias21_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4'
)


# stream PhysicsSpecialZeroBias11

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias11_datasetSpecialZeroBias22_selector
streamPhysicsSpecialZeroBias11_datasetSpecialZeroBias22_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias11_datasetSpecialZeroBias22_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias11_datasetSpecialZeroBias22_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias11_datasetSpecialZeroBias22_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias11_datasetSpecialZeroBias23_selector
streamPhysicsSpecialZeroBias11_datasetSpecialZeroBias23_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias11_datasetSpecialZeroBias23_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias11_datasetSpecialZeroBias23_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias11_datasetSpecialZeroBias23_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4'
)


# stream PhysicsSpecialZeroBias12

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias12_datasetSpecialZeroBias24_selector
streamPhysicsSpecialZeroBias12_datasetSpecialZeroBias24_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias12_datasetSpecialZeroBias24_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias12_datasetSpecialZeroBias24_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias12_datasetSpecialZeroBias24_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias12_datasetSpecialZeroBias25_selector
streamPhysicsSpecialZeroBias12_datasetSpecialZeroBias25_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias12_datasetSpecialZeroBias25_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias12_datasetSpecialZeroBias25_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias12_datasetSpecialZeroBias25_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4'
)


# stream PhysicsSpecialZeroBias13

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias13_datasetSpecialZeroBias26_selector
streamPhysicsSpecialZeroBias13_datasetSpecialZeroBias26_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias13_datasetSpecialZeroBias26_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias13_datasetSpecialZeroBias26_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias13_datasetSpecialZeroBias26_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias13_datasetSpecialZeroBias27_selector
streamPhysicsSpecialZeroBias13_datasetSpecialZeroBias27_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias13_datasetSpecialZeroBias27_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias13_datasetSpecialZeroBias27_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias13_datasetSpecialZeroBias27_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4'
)


# stream PhysicsSpecialZeroBias14

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias14_datasetSpecialZeroBias28_selector
streamPhysicsSpecialZeroBias14_datasetSpecialZeroBias28_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias14_datasetSpecialZeroBias28_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias14_datasetSpecialZeroBias28_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias14_datasetSpecialZeroBias28_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias14_datasetSpecialZeroBias29_selector
streamPhysicsSpecialZeroBias14_datasetSpecialZeroBias29_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias14_datasetSpecialZeroBias29_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias14_datasetSpecialZeroBias29_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias14_datasetSpecialZeroBias29_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4'
)


# stream PhysicsSpecialZeroBias15

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias15_datasetSpecialZeroBias30_selector
streamPhysicsSpecialZeroBias15_datasetSpecialZeroBias30_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias15_datasetSpecialZeroBias30_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias15_datasetSpecialZeroBias30_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias15_datasetSpecialZeroBias30_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias15_datasetSpecialZeroBias31_selector
streamPhysicsSpecialZeroBias15_datasetSpecialZeroBias31_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias15_datasetSpecialZeroBias31_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias15_datasetSpecialZeroBias31_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias15_datasetSpecialZeroBias31_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4'
)


# stream PhysicsSpecialZeroBias2

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias2_datasetSpecialZeroBias4_selector
streamPhysicsSpecialZeroBias2_datasetSpecialZeroBias4_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias2_datasetSpecialZeroBias4_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias2_datasetSpecialZeroBias4_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias2_datasetSpecialZeroBias4_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias2_datasetSpecialZeroBias5_selector
streamPhysicsSpecialZeroBias2_datasetSpecialZeroBias5_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias2_datasetSpecialZeroBias5_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias2_datasetSpecialZeroBias5_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias2_datasetSpecialZeroBias5_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4'
)


# stream PhysicsSpecialZeroBias3

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias3_datasetSpecialZeroBias6_selector
streamPhysicsSpecialZeroBias3_datasetSpecialZeroBias6_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias3_datasetSpecialZeroBias6_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias3_datasetSpecialZeroBias6_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias3_datasetSpecialZeroBias6_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias3_datasetSpecialZeroBias7_selector
streamPhysicsSpecialZeroBias3_datasetSpecialZeroBias7_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias3_datasetSpecialZeroBias7_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias3_datasetSpecialZeroBias7_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias3_datasetSpecialZeroBias7_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4'
)


# stream PhysicsSpecialZeroBias4

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias4_datasetSpecialZeroBias8_selector
streamPhysicsSpecialZeroBias4_datasetSpecialZeroBias8_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias4_datasetSpecialZeroBias8_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias4_datasetSpecialZeroBias8_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias4_datasetSpecialZeroBias8_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias4_datasetSpecialZeroBias9_selector
streamPhysicsSpecialZeroBias4_datasetSpecialZeroBias9_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias4_datasetSpecialZeroBias9_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias4_datasetSpecialZeroBias9_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias4_datasetSpecialZeroBias9_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4'
)


# stream PhysicsSpecialZeroBias5

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias5_datasetSpecialZeroBias10_selector
streamPhysicsSpecialZeroBias5_datasetSpecialZeroBias10_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias5_datasetSpecialZeroBias10_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias5_datasetSpecialZeroBias10_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias5_datasetSpecialZeroBias10_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias5_datasetSpecialZeroBias11_selector
streamPhysicsSpecialZeroBias5_datasetSpecialZeroBias11_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias5_datasetSpecialZeroBias11_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias5_datasetSpecialZeroBias11_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias5_datasetSpecialZeroBias11_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4'
)


# stream PhysicsSpecialZeroBias6

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias6_datasetSpecialZeroBias12_selector
streamPhysicsSpecialZeroBias6_datasetSpecialZeroBias12_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias6_datasetSpecialZeroBias12_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias6_datasetSpecialZeroBias12_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias6_datasetSpecialZeroBias12_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias6_datasetSpecialZeroBias13_selector
streamPhysicsSpecialZeroBias6_datasetSpecialZeroBias13_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias6_datasetSpecialZeroBias13_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias6_datasetSpecialZeroBias13_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias6_datasetSpecialZeroBias13_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4'
)


# stream PhysicsSpecialZeroBias7

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias7_datasetSpecialZeroBias14_selector
streamPhysicsSpecialZeroBias7_datasetSpecialZeroBias14_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias7_datasetSpecialZeroBias14_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias7_datasetSpecialZeroBias14_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias7_datasetSpecialZeroBias14_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias7_datasetSpecialZeroBias15_selector
streamPhysicsSpecialZeroBias7_datasetSpecialZeroBias15_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias7_datasetSpecialZeroBias15_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias7_datasetSpecialZeroBias15_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias7_datasetSpecialZeroBias15_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4'
)


# stream PhysicsSpecialZeroBias8

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias8_datasetSpecialZeroBias16_selector
streamPhysicsSpecialZeroBias8_datasetSpecialZeroBias16_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias8_datasetSpecialZeroBias16_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias8_datasetSpecialZeroBias16_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias8_datasetSpecialZeroBias16_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias8_datasetSpecialZeroBias17_selector
streamPhysicsSpecialZeroBias8_datasetSpecialZeroBias17_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias8_datasetSpecialZeroBias17_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias8_datasetSpecialZeroBias17_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias8_datasetSpecialZeroBias17_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4'
)


# stream PhysicsSpecialZeroBias9

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias9_datasetSpecialZeroBias18_selector
streamPhysicsSpecialZeroBias9_datasetSpecialZeroBias18_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias9_datasetSpecialZeroBias18_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias9_datasetSpecialZeroBias18_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias9_datasetSpecialZeroBias18_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsSpecialZeroBias9_datasetSpecialZeroBias19_selector
streamPhysicsSpecialZeroBias9_datasetSpecialZeroBias19_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsSpecialZeroBias9_datasetSpecialZeroBias19_selector.l1tResults = cms.InputTag('')
streamPhysicsSpecialZeroBias9_datasetSpecialZeroBias19_selector.throw      = cms.bool(False)
streamPhysicsSpecialZeroBias9_datasetSpecialZeroBias19_selector.triggerConditions = cms.vstring(
    'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4'
)


# stream PhysicsVRRandom0

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsVRRandom0_datasetVRRandom0_selector
streamPhysicsVRRandom0_datasetVRRandom0_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsVRRandom0_datasetVRRandom0_selector.l1tResults = cms.InputTag('')
streamPhysicsVRRandom0_datasetVRRandom0_selector.throw      = cms.bool(False)
streamPhysicsVRRandom0_datasetVRRandom0_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsVRRandom0_datasetVRRandom1_selector
streamPhysicsVRRandom0_datasetVRRandom1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsVRRandom0_datasetVRRandom1_selector.l1tResults = cms.InputTag('')
streamPhysicsVRRandom0_datasetVRRandom1_selector.throw      = cms.bool(False)
streamPhysicsVRRandom0_datasetVRRandom1_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')


# stream PhysicsVRRandom1

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsVRRandom1_datasetVRRandom2_selector
streamPhysicsVRRandom1_datasetVRRandom2_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsVRRandom1_datasetVRRandom2_selector.l1tResults = cms.InputTag('')
streamPhysicsVRRandom1_datasetVRRandom2_selector.throw      = cms.bool(False)
streamPhysicsVRRandom1_datasetVRRandom2_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsVRRandom1_datasetVRRandom3_selector
streamPhysicsVRRandom1_datasetVRRandom3_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsVRRandom1_datasetVRRandom3_selector.l1tResults = cms.InputTag('')
streamPhysicsVRRandom1_datasetVRRandom3_selector.throw      = cms.bool(False)
streamPhysicsVRRandom1_datasetVRRandom3_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')


# stream PhysicsVRRandom2

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsVRRandom2_datasetVRRandom4_selector
streamPhysicsVRRandom2_datasetVRRandom4_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsVRRandom2_datasetVRRandom4_selector.l1tResults = cms.InputTag('')
streamPhysicsVRRandom2_datasetVRRandom4_selector.throw      = cms.bool(False)
streamPhysicsVRRandom2_datasetVRRandom4_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsVRRandom2_datasetVRRandom5_selector
streamPhysicsVRRandom2_datasetVRRandom5_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsVRRandom2_datasetVRRandom5_selector.l1tResults = cms.InputTag('')
streamPhysicsVRRandom2_datasetVRRandom5_selector.throw      = cms.bool(False)
streamPhysicsVRRandom2_datasetVRRandom5_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')


# stream PhysicsVRRandom3

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsVRRandom3_datasetVRRandom6_selector
streamPhysicsVRRandom3_datasetVRRandom6_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsVRRandom3_datasetVRRandom6_selector.l1tResults = cms.InputTag('')
streamPhysicsVRRandom3_datasetVRRandom6_selector.throw      = cms.bool(False)
streamPhysicsVRRandom3_datasetVRRandom6_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsVRRandom3_datasetVRRandom7_selector
streamPhysicsVRRandom3_datasetVRRandom7_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsVRRandom3_datasetVRRandom7_selector.l1tResults = cms.InputTag('')
streamPhysicsVRRandom3_datasetVRRandom7_selector.throw      = cms.bool(False)
streamPhysicsVRRandom3_datasetVRRandom7_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')


# stream PhysicsVRRandom4

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsVRRandom4_datasetVRRandom8_selector
streamPhysicsVRRandom4_datasetVRRandom8_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsVRRandom4_datasetVRRandom8_selector.l1tResults = cms.InputTag('')
streamPhysicsVRRandom4_datasetVRRandom8_selector.throw      = cms.bool(False)
streamPhysicsVRRandom4_datasetVRRandom8_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsVRRandom4_datasetVRRandom9_selector
streamPhysicsVRRandom4_datasetVRRandom9_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsVRRandom4_datasetVRRandom9_selector.l1tResults = cms.InputTag('')
streamPhysicsVRRandom4_datasetVRRandom9_selector.throw      = cms.bool(False)
streamPhysicsVRRandom4_datasetVRRandom9_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')


# stream PhysicsVRRandom5

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsVRRandom5_datasetVRRandom10_selector
streamPhysicsVRRandom5_datasetVRRandom10_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsVRRandom5_datasetVRRandom10_selector.l1tResults = cms.InputTag('')
streamPhysicsVRRandom5_datasetVRRandom10_selector.throw      = cms.bool(False)
streamPhysicsVRRandom5_datasetVRRandom10_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsVRRandom5_datasetVRRandom11_selector
streamPhysicsVRRandom5_datasetVRRandom11_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsVRRandom5_datasetVRRandom11_selector.l1tResults = cms.InputTag('')
streamPhysicsVRRandom5_datasetVRRandom11_selector.throw      = cms.bool(False)
streamPhysicsVRRandom5_datasetVRRandom11_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')


# stream PhysicsVRRandom6

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsVRRandom6_datasetVRRandom12_selector
streamPhysicsVRRandom6_datasetVRRandom12_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsVRRandom6_datasetVRRandom12_selector.l1tResults = cms.InputTag('')
streamPhysicsVRRandom6_datasetVRRandom12_selector.throw      = cms.bool(False)
streamPhysicsVRRandom6_datasetVRRandom12_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsVRRandom6_datasetVRRandom13_selector
streamPhysicsVRRandom6_datasetVRRandom13_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsVRRandom6_datasetVRRandom13_selector.l1tResults = cms.InputTag('')
streamPhysicsVRRandom6_datasetVRRandom13_selector.throw      = cms.bool(False)
streamPhysicsVRRandom6_datasetVRRandom13_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')


# stream PhysicsVRRandom7

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsVRRandom7_datasetVRRandom14_selector
streamPhysicsVRRandom7_datasetVRRandom14_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsVRRandom7_datasetVRRandom14_selector.l1tResults = cms.InputTag('')
streamPhysicsVRRandom7_datasetVRRandom14_selector.throw      = cms.bool(False)
streamPhysicsVRRandom7_datasetVRRandom14_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsVRRandom7_datasetVRRandom15_selector
streamPhysicsVRRandom7_datasetVRRandom15_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsVRRandom7_datasetVRRandom15_selector.l1tResults = cms.InputTag('')
streamPhysicsVRRandom7_datasetVRRandom15_selector.throw      = cms.bool(False)
streamPhysicsVRRandom7_datasetVRRandom15_selector.triggerConditions = cms.vstring('HLT_Random_HighRate_v1')

