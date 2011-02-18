# /dev/CMSSW_3_11_1/GRun/V31

import FWCore.ParameterSet.Config as cms

# dump of the Stream A Datasets defined in the HLT table

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetA_selector
streamA_datasetA_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetA_selector.l1tResults = cms.InputTag('')
streamA_datasetA_selector.throw      = cms.bool(False)
streamA_datasetA_selector.triggerConditions = cms.vstring('HLT_BTagMu_DiJet20_Mu5_v1', 
    'HLT_BTagMu_DiJet60_Mu7_v1', 
    'HLT_BTagMu_DiJet80_Mu9_v1', 
    'HLT_CentralJet80_MET100_v1', 
    'HLT_CentralJet80_MET160_v1', 
    'HLT_CentralJet80_MET65_v1', 
    'HLT_CentralJet80_MET80_v1', 
    'HLT_DiJet100_PT100_v1', 
    'HLT_DiJet130_PT130_v1', 
    'HLT_DiJet60_MET45_v1', 
    'HLT_DiJet70_PT70_v1', 
    'HLT_DoubleMu3_v3', 
    'HLT_DoubleMu6_v1', 
    'HLT_DoubleMu7_v1', 
    'HLT_HT160_v1', 
    'HLT_HT240_v1', 
    'HLT_HT260_MHT60_v1', 
    'HLT_HT300_MHT75_v1', 
    'HLT_HT300_v1', 
    'HLT_HT360_v1', 
    'HLT_HT440_v1', 
    'HLT_HT520_v1', 
    'HLT_IsoMu12_v1', 
    'HLT_IsoMu15_v5', 
    'HLT_IsoMu17_v5', 
    'HLT_IsoMu30_v1', 
    'HLT_MET100_v1', 
    'HLT_MET120_v1', 
    'HLT_MET200_v1', 
    'HLT_MR100_v1', 
    'HLT_Meff440_v1', 
    'HLT_Meff520_v1', 
    'HLT_Meff640_v1', 
    'HLT_Mu12_v1', 
    'HLT_Mu15_v2', 
    'HLT_Mu20_v1', 
    'HLT_Mu24_v1', 
    'HLT_Mu30_v1', 
    'HLT_Mu3_v2', 
    'HLT_Mu5_v2', 
    'HLT_QuadJet40_v1', 
    'HLT_QuadJet50_BTagIP_v1', 
    'HLT_QuadJet50_Jet40_v1', 
    'HLT_QuadJet60_v1', 
    'HLT_QuadJet65_v1', 
    'HLT_QuadJet70_v1', 
    'HLT_R032_MR100_v1', 
    'HLT_R032_v1', 
    'HLT_R035_MR100_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHcalNZS_selector
streamA_datasetHcalNZS_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHcalNZS_selector.l1tResults = cms.InputTag('')
streamA_datasetHcalNZS_selector.throw      = cms.bool(False)
streamA_datasetHcalNZS_selector.triggerConditions = cms.vstring('HLT_HcalNZS_v2', 
    'HLT_HcalPhiSym_v2')

