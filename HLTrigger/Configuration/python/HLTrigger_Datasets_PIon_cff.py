# /dev/CMSSW_15_0_0/PIon

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
streamPhysicsCommissioning_datasetHLTPhysics_selector.triggerConditions = cms.vstring('HLT_Physics_v15')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetHcalNZS_selector
streamPhysicsCommissioning_datasetHcalNZS_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetHcalNZS_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetHcalNZS_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetHcalNZS_selector.triggerConditions = cms.vstring(
    'HLT_HcalNZS_v22',
    'HLT_HcalPhiSym_v24'
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


# stream PhysicsIonPhysics0

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics0_datasetIonPhysics0_selector
streamPhysicsIonPhysics0_datasetIonPhysics0_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics0_datasetIonPhysics0_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics0_datasetIonPhysics0_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics0_datasetIonPhysics0_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics1

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics1_datasetIonPhysics1_selector
streamPhysicsIonPhysics1_datasetIonPhysics1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics1_datasetIonPhysics1_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics1_datasetIonPhysics1_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics1_datasetIonPhysics1_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics10

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics10_datasetIonPhysics10_selector
streamPhysicsIonPhysics10_datasetIonPhysics10_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics10_datasetIonPhysics10_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics10_datasetIonPhysics10_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics10_datasetIonPhysics10_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics11

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics11_datasetIonPhysics11_selector
streamPhysicsIonPhysics11_datasetIonPhysics11_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics11_datasetIonPhysics11_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics11_datasetIonPhysics11_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics11_datasetIonPhysics11_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics12

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics12_datasetIonPhysics12_selector
streamPhysicsIonPhysics12_datasetIonPhysics12_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics12_datasetIonPhysics12_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics12_datasetIonPhysics12_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics12_datasetIonPhysics12_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics13

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics13_datasetIonPhysics13_selector
streamPhysicsIonPhysics13_datasetIonPhysics13_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics13_datasetIonPhysics13_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics13_datasetIonPhysics13_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics13_datasetIonPhysics13_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics14

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics14_datasetIonPhysics14_selector
streamPhysicsIonPhysics14_datasetIonPhysics14_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics14_datasetIonPhysics14_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics14_datasetIonPhysics14_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics14_datasetIonPhysics14_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics15

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics15_datasetIonPhysics15_selector
streamPhysicsIonPhysics15_datasetIonPhysics15_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics15_datasetIonPhysics15_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics15_datasetIonPhysics15_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics15_datasetIonPhysics15_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics16

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics16_datasetIonPhysics16_selector
streamPhysicsIonPhysics16_datasetIonPhysics16_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics16_datasetIonPhysics16_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics16_datasetIonPhysics16_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics16_datasetIonPhysics16_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics17

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics17_datasetIonPhysics17_selector
streamPhysicsIonPhysics17_datasetIonPhysics17_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics17_datasetIonPhysics17_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics17_datasetIonPhysics17_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics17_datasetIonPhysics17_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics18

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics18_datasetIonPhysics18_selector
streamPhysicsIonPhysics18_datasetIonPhysics18_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics18_datasetIonPhysics18_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics18_datasetIonPhysics18_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics18_datasetIonPhysics18_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics19

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics19_datasetIonPhysics19_selector
streamPhysicsIonPhysics19_datasetIonPhysics19_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics19_datasetIonPhysics19_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics19_datasetIonPhysics19_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics19_datasetIonPhysics19_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics2

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics2_datasetIonPhysics2_selector
streamPhysicsIonPhysics2_datasetIonPhysics2_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics2_datasetIonPhysics2_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics2_datasetIonPhysics2_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics2_datasetIonPhysics2_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics20

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics20_datasetIonPhysics20_selector
streamPhysicsIonPhysics20_datasetIonPhysics20_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics20_datasetIonPhysics20_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics20_datasetIonPhysics20_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics20_datasetIonPhysics20_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics21

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics21_datasetIonPhysics21_selector
streamPhysicsIonPhysics21_datasetIonPhysics21_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics21_datasetIonPhysics21_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics21_datasetIonPhysics21_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics21_datasetIonPhysics21_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics22

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics22_datasetIonPhysics22_selector
streamPhysicsIonPhysics22_datasetIonPhysics22_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics22_datasetIonPhysics22_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics22_datasetIonPhysics22_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics22_datasetIonPhysics22_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics23

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics23_datasetIonPhysics23_selector
streamPhysicsIonPhysics23_datasetIonPhysics23_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics23_datasetIonPhysics23_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics23_datasetIonPhysics23_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics23_datasetIonPhysics23_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics24

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics24_datasetIonPhysics24_selector
streamPhysicsIonPhysics24_datasetIonPhysics24_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics24_datasetIonPhysics24_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics24_datasetIonPhysics24_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics24_datasetIonPhysics24_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics25

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics25_datasetIonPhysics25_selector
streamPhysicsIonPhysics25_datasetIonPhysics25_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics25_datasetIonPhysics25_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics25_datasetIonPhysics25_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics25_datasetIonPhysics25_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics26

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics26_datasetIonPhysics26_selector
streamPhysicsIonPhysics26_datasetIonPhysics26_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics26_datasetIonPhysics26_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics26_datasetIonPhysics26_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics26_datasetIonPhysics26_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics27

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics27_datasetIonPhysics27_selector
streamPhysicsIonPhysics27_datasetIonPhysics27_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics27_datasetIonPhysics27_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics27_datasetIonPhysics27_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics27_datasetIonPhysics27_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics28

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics28_datasetIonPhysics28_selector
streamPhysicsIonPhysics28_datasetIonPhysics28_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics28_datasetIonPhysics28_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics28_datasetIonPhysics28_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics28_datasetIonPhysics28_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics29

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics29_datasetIonPhysics29_selector
streamPhysicsIonPhysics29_datasetIonPhysics29_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics29_datasetIonPhysics29_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics29_datasetIonPhysics29_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics29_datasetIonPhysics29_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics3

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics3_datasetIonPhysics3_selector
streamPhysicsIonPhysics3_datasetIonPhysics3_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics3_datasetIonPhysics3_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics3_datasetIonPhysics3_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics3_datasetIonPhysics3_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics30

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics30_datasetIonPhysics30_selector
streamPhysicsIonPhysics30_datasetIonPhysics30_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics30_datasetIonPhysics30_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics30_datasetIonPhysics30_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics30_datasetIonPhysics30_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics31

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics31_datasetIonPhysics31_selector
streamPhysicsIonPhysics31_datasetIonPhysics31_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics31_datasetIonPhysics31_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics31_datasetIonPhysics31_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics31_datasetIonPhysics31_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics32

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics32_datasetIonPhysics32_selector
streamPhysicsIonPhysics32_datasetIonPhysics32_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics32_datasetIonPhysics32_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics32_datasetIonPhysics32_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics32_datasetIonPhysics32_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics33

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics33_datasetIonPhysics33_selector
streamPhysicsIonPhysics33_datasetIonPhysics33_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics33_datasetIonPhysics33_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics33_datasetIonPhysics33_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics33_datasetIonPhysics33_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics34

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics34_datasetIonPhysics34_selector
streamPhysicsIonPhysics34_datasetIonPhysics34_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics34_datasetIonPhysics34_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics34_datasetIonPhysics34_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics34_datasetIonPhysics34_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics35

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics35_datasetIonPhysics35_selector
streamPhysicsIonPhysics35_datasetIonPhysics35_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics35_datasetIonPhysics35_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics35_datasetIonPhysics35_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics35_datasetIonPhysics35_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics36

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics36_datasetIonPhysics36_selector
streamPhysicsIonPhysics36_datasetIonPhysics36_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics36_datasetIonPhysics36_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics36_datasetIonPhysics36_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics36_datasetIonPhysics36_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics37

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics37_datasetIonPhysics37_selector
streamPhysicsIonPhysics37_datasetIonPhysics37_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics37_datasetIonPhysics37_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics37_datasetIonPhysics37_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics37_datasetIonPhysics37_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics38

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics38_datasetIonPhysics38_selector
streamPhysicsIonPhysics38_datasetIonPhysics38_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics38_datasetIonPhysics38_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics38_datasetIonPhysics38_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics38_datasetIonPhysics38_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics39

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics39_datasetIonPhysics39_selector
streamPhysicsIonPhysics39_datasetIonPhysics39_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics39_datasetIonPhysics39_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics39_datasetIonPhysics39_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics39_datasetIonPhysics39_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics4

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics4_datasetIonPhysics4_selector
streamPhysicsIonPhysics4_datasetIonPhysics4_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics4_datasetIonPhysics4_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics4_datasetIonPhysics4_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics4_datasetIonPhysics4_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics40

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics40_datasetIonPhysics40_selector
streamPhysicsIonPhysics40_datasetIonPhysics40_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics40_datasetIonPhysics40_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics40_datasetIonPhysics40_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics40_datasetIonPhysics40_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics41

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics41_datasetIonPhysics41_selector
streamPhysicsIonPhysics41_datasetIonPhysics41_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics41_datasetIonPhysics41_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics41_datasetIonPhysics41_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics41_datasetIonPhysics41_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics42

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics42_datasetIonPhysics42_selector
streamPhysicsIonPhysics42_datasetIonPhysics42_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics42_datasetIonPhysics42_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics42_datasetIonPhysics42_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics42_datasetIonPhysics42_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics43

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics43_datasetIonPhysics43_selector
streamPhysicsIonPhysics43_datasetIonPhysics43_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics43_datasetIonPhysics43_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics43_datasetIonPhysics43_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics43_datasetIonPhysics43_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics44

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics44_datasetIonPhysics44_selector
streamPhysicsIonPhysics44_datasetIonPhysics44_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics44_datasetIonPhysics44_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics44_datasetIonPhysics44_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics44_datasetIonPhysics44_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics45

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics45_datasetIonPhysics45_selector
streamPhysicsIonPhysics45_datasetIonPhysics45_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics45_datasetIonPhysics45_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics45_datasetIonPhysics45_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics45_datasetIonPhysics45_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics46

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics46_datasetIonPhysics46_selector
streamPhysicsIonPhysics46_datasetIonPhysics46_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics46_datasetIonPhysics46_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics46_datasetIonPhysics46_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics46_datasetIonPhysics46_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics47

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics47_datasetIonPhysics47_selector
streamPhysicsIonPhysics47_datasetIonPhysics47_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics47_datasetIonPhysics47_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics47_datasetIonPhysics47_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics47_datasetIonPhysics47_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics48

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics48_datasetIonPhysics48_selector
streamPhysicsIonPhysics48_datasetIonPhysics48_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics48_datasetIonPhysics48_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics48_datasetIonPhysics48_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics48_datasetIonPhysics48_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics49

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics49_datasetIonPhysics49_selector
streamPhysicsIonPhysics49_datasetIonPhysics49_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics49_datasetIonPhysics49_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics49_datasetIonPhysics49_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics49_datasetIonPhysics49_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics5

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics5_datasetIonPhysics5_selector
streamPhysicsIonPhysics5_datasetIonPhysics5_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics5_datasetIonPhysics5_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics5_datasetIonPhysics5_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics5_datasetIonPhysics5_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics50

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics50_datasetIonPhysics50_selector
streamPhysicsIonPhysics50_datasetIonPhysics50_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics50_datasetIonPhysics50_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics50_datasetIonPhysics50_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics50_datasetIonPhysics50_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics51

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics51_datasetIonPhysics51_selector
streamPhysicsIonPhysics51_datasetIonPhysics51_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics51_datasetIonPhysics51_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics51_datasetIonPhysics51_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics51_datasetIonPhysics51_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics52

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics52_datasetIonPhysics52_selector
streamPhysicsIonPhysics52_datasetIonPhysics52_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics52_datasetIonPhysics52_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics52_datasetIonPhysics52_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics52_datasetIonPhysics52_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics53

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics53_datasetIonPhysics53_selector
streamPhysicsIonPhysics53_datasetIonPhysics53_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics53_datasetIonPhysics53_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics53_datasetIonPhysics53_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics53_datasetIonPhysics53_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics54

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics54_datasetIonPhysics54_selector
streamPhysicsIonPhysics54_datasetIonPhysics54_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics54_datasetIonPhysics54_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics54_datasetIonPhysics54_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics54_datasetIonPhysics54_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics55

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics55_datasetIonPhysics55_selector
streamPhysicsIonPhysics55_datasetIonPhysics55_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics55_datasetIonPhysics55_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics55_datasetIonPhysics55_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics55_datasetIonPhysics55_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics56

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics56_datasetIonPhysics56_selector
streamPhysicsIonPhysics56_datasetIonPhysics56_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics56_datasetIonPhysics56_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics56_datasetIonPhysics56_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics56_datasetIonPhysics56_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics57

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics57_datasetIonPhysics57_selector
streamPhysicsIonPhysics57_datasetIonPhysics57_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics57_datasetIonPhysics57_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics57_datasetIonPhysics57_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics57_datasetIonPhysics57_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics58

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics58_datasetIonPhysics58_selector
streamPhysicsIonPhysics58_datasetIonPhysics58_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics58_datasetIonPhysics58_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics58_datasetIonPhysics58_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics58_datasetIonPhysics58_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics59

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics59_datasetIonPhysics59_selector
streamPhysicsIonPhysics59_datasetIonPhysics59_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics59_datasetIonPhysics59_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics59_datasetIonPhysics59_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics59_datasetIonPhysics59_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics6

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics6_datasetIonPhysics6_selector
streamPhysicsIonPhysics6_datasetIonPhysics6_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics6_datasetIonPhysics6_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics6_datasetIonPhysics6_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics6_datasetIonPhysics6_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics7

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics7_datasetIonPhysics7_selector
streamPhysicsIonPhysics7_datasetIonPhysics7_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics7_datasetIonPhysics7_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics7_datasetIonPhysics7_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics7_datasetIonPhysics7_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics8

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics8_datasetIonPhysics8_selector
streamPhysicsIonPhysics8_datasetIonPhysics8_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics8_datasetIonPhysics8_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics8_datasetIonPhysics8_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics8_datasetIonPhysics8_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
    'HLT_OxyZeroBias_v1'
)


# stream PhysicsIonPhysics9

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsIonPhysics9_datasetIonPhysics9_selector
streamPhysicsIonPhysics9_datasetIonPhysics9_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsIonPhysics9_datasetIonPhysics9_selector.l1tResults = cms.InputTag('')
streamPhysicsIonPhysics9_datasetIonPhysics9_selector.throw      = cms.bool(False)
streamPhysicsIonPhysics9_datasetIonPhysics9_selector.triggerConditions = cms.vstring(
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
    'HLT_OxyZeroBias_MinPixelCluster400_v1',
    'HLT_OxyZeroBias_SinglePixelTrackLowPt_MaxPixelCluster400_v1',
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

