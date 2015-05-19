# getDatasets.py

import FWCore.ParameterSet.Config as cms


# dump of the Stream A Datasets defined in the HLT table as Stream A Datasets

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetInitialPD_selector
streamA_datasetInitialPD_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetInitialPD_selector.l1tResults = cms.InputTag('')
streamA_datasetInitialPD_selector.throw      = cms.bool(False)
streamA_datasetInitialPD_selector.triggerConditions = cms.vstring('HLT_Ele27_eta2p1_WPLoose_Gsf_v1', 
    'HLT_Mu50_v2', 
    'HLT_PFJet260_v2', 
    'HLT_Physics_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetTemplates_selector
streamA_datasetTemplates_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetTemplates_selector.l1tResults = cms.InputTag('')
streamA_datasetTemplates_selector.throw      = cms.bool(False)
streamA_datasetTemplates_selector.triggerConditions = cms.vstring('HLT_CaloJet260_v2', 
    'HLT_Photon20_CaloIdVL_IsoL_v2')

