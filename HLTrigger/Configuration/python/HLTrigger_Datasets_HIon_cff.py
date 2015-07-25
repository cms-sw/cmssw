# getDatasets.py

import FWCore.ParameterSet.Config as cms


# dump of the Stream PhysicsEGammaCommissioning Datasets defined in the HLT table as Stream A Datasets

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHLTPhysics_selector
streamA_datasetHLTPhysics_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHLTPhysics_selector.l1tResults = cms.InputTag('')
streamA_datasetHLTPhysics_selector.throw      = cms.bool(False)
streamA_datasetHLTPhysics_selector.triggerConditions = cms.vstring('HLT_Physics_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetInitialPDForHI_selector
streamA_datasetInitialPDForHI_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetInitialPDForHI_selector.l1tResults = cms.InputTag('')
streamA_datasetInitialPDForHI_selector.throw      = cms.bool(False)
streamA_datasetInitialPDForHI_selector.triggerConditions = cms.vstring('HLT_HIL1DoubleMu0_HighQ_v2', 
    'HLT_HIL2DoubleMu0_NHitQ_v2', 
    'HLT_HIL2DoubleMu0_v2', 
    'HLT_HIL2DoubleMu3_v2', 
    'HLT_HIL2Mu15_v2', 
    'HLT_HIL2Mu3_NHitQ_v2', 
    'HLT_HIL2Mu3_v2', 
    'HLT_HIL2Mu7_v2', 
    'HLT_HIL3DoubleMuOpen_OS_NoCowboy_v2', 
    'HLT_HIL3DoubleMuOpen_OS_v2', 
    'HLT_HIL3DoubleMuOpen_SS_v2', 
    'HLT_HIL3DoubleMuOpen_v2', 
    'HLT_HIL3Mu3_v2')

