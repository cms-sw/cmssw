# getDatasets.py

import FWCore.ParameterSet.Config as cms


# dump of the Stream A Datasets defined in the HLT table as Stream A Datasets

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetInitialPD_selector
streamA_datasetInitialPD_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetInitialPD_selector.l1tResults = cms.InputTag('')
streamA_datasetInitialPD_selector.throw      = cms.bool(False)
streamA_datasetInitialPD_selector.triggerConditions = cms.vstring('HLT_CaloJet260_v1', 
    'HLT_Ele27_WP85_Gsf_v1', 
    'HLT_Mu40_v1', 
    'HLT_PFJet260_v1', 
    'HLT_Photon20_CaloIdVL_IsoL_v1', 
    'HLT_Physics_v1')

