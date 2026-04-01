# /dev/CMSSW_15_1_0/PIon

import FWCore.ParameterSet.Config as cms


# stream PhysicsCommissioning

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetEmptyBX_selector
streamPhysicsCommissioning_datasetEmptyBX_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetEmptyBX_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetEmptyBX_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetEmptyBX_selector.triggerConditions = cms.vstring(
    'HLT_HIL1NotBptxORForPPRef_v10',
    'HLT_HIL1UnpairedBunchBptxMinusForPPRef_v10',
    'HLT_HIL1UnpairedBunchBptxPlusForPPRef_v10'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetHLTPhysics_selector
streamPhysicsCommissioning_datasetHLTPhysics_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetHLTPhysics_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetHLTPhysics_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetHLTPhysics_selector.triggerConditions = cms.vstring('HLT_Physics_v16')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetHcalNZS_selector
streamPhysicsCommissioning_datasetHcalNZS_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetHcalNZS_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetHcalNZS_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetHcalNZS_selector.triggerConditions = cms.vstring(
    'HLT_HcalNZS_v23',
    'HLT_HcalPhiSym_v25'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetNoBPTX_selector
streamPhysicsCommissioning_datasetNoBPTX_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetNoBPTX_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetNoBPTX_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetNoBPTX_selector.triggerConditions = cms.vstring(
    'HLT_CDC_L2cosmic_10_er1p0_v11',
    'HLT_CDC_L2cosmic_5p5_er1p0_v11'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetZeroBias_selector
streamPhysicsCommissioning_datasetZeroBias_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetZeroBias_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetZeroBias_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetZeroBias_selector.triggerConditions = cms.vstring(
    'HLT_Random_v3',
    'HLT_ZeroBias_FirstCollisionAfterAbortGap_v13',
    'HLT_ZeroBias_v14'
)


# stream PhysicsIonPhysics

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics_datasetIonPhysics_selector
streamPhysicsIonPhysics_datasetIonPhysics_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics_datasetIonPhysics_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics_datasetIonPhysics_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics_datasetIonPhysics_selector.triggerConditions = cms.vstring(
    'HLT_MinimumBiasHF_AND_BptxAND_v1',
    'HLT_MinimumBiasHF_OR_BptxAND_v1',
    'HLT_MinimumBiasZDC1n_OR_BptxAND_v1',
    'HLT_MinimumBiasZDC1n_OR_MinimumBiasHF_AND_BptxAND_v1',
    'HLT_MinimumBiasZDC1n_OR_MinimumBiasHF_OR_BptxAND_v1',
    'HLT_MinimumBiasZDC1n_OR_OR_MinimumBiasHF_OR_BptxAND_v1',
    'HLT_MinimumBiasZDC_Asym_MinimumBiasHF_OR_BptxAND_v1',
    'HLT_OxyDoubleEG2_NotMBHF2AND_v1',
    'HLT_OxyDoubleEG5_NotMBHF2AND_v1',
    'HLT_OxyL1CentralityGate_v1',
    'HLT_OxyL1DoubleMu0_v1',
    'HLT_OxyL1DoubleMuOpen_v1',
    'HLT_OxyL1SingleEG10_v1',
    'HLT_OxyL1SingleEG15_v1',
    'HLT_OxyL1SingleEG15er2p1_v1',
    'HLT_OxyL1SingleEG15er2p5_v1',
    'HLT_OxyL1SingleEG21_v1',
    'HLT_OxyL1SingleJet20_v1',
    'HLT_OxyL1SingleJet28_v1',
    'HLT_OxyL1SingleJet35_v1',
    'HLT_OxyL1SingleJet44_v1',
    'HLT_OxyL1SingleJet60_v1',
    'HLT_OxyL1SingleMu0_v1',
    'HLT_OxyL1SingleMu3_v1',
    'HLT_OxyL1SingleMu5_v1',
    'HLT_OxyL1SingleMu7_v1',
    'HLT_OxyL1SingleMuOpen_v1',
    'HLT_OxyNotMBHF2_v1',
    'HLT_OxySingleEG2_NotMBHF2AND_ZDC1nOR_v1',
    'HLT_OxySingleEG3_NotMBHF2AND_v1',
    'HLT_OxySingleEG3_NotMBHF2OR_v1',
    'HLT_OxySingleEG5_NotMBHF2AND_v1',
    'HLT_OxySingleJet16_ZDC1nAsymXOR_NotPreBptx_v1',
    'HLT_OxySingleJet16_ZDC1nAsymXOR_v1',
    'HLT_OxySingleJet16_ZDC1nXOR_v1',
    'HLT_OxySingleJet24_ZDC1nAsymXOR_NotPreBptx_v1',
    'HLT_OxySingleJet24_ZDC1nAsymXOR_v1',
    'HLT_OxySingleJet24_ZDC1nXOR_v1',
    'HLT_OxySingleJet8_ZDC1nAsymXOR_NotPreBptx_v1',
    'HLT_OxySingleJet8_ZDC1nAsymXOR_v1',
    'HLT_OxySingleJet8_ZDC1nXOR_v1',
    'HLT_OxySingleMuCosmic_NotMBHF2AND_v1',
    'HLT_OxySingleMuOpen_NotMBHF2AND_v1',
    'HLT_OxySingleMuOpen_NotMBHF2OR_v1',
    'HLT_OxySingleMuOpen_OR_SingleMuCosmic_EMTF_BptxAND_v1',
    'HLT_OxySingleMuOpen_OR_SingleMuCosmic_EMTF_NotMBHF2AND_v1',
    'HLT_OxyZDC1nOR_v1',
    'HLT_OxyZeroBias_MinPixelCluster400_v2',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v2',
    'HLT_OxyZeroBias_v1'
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

