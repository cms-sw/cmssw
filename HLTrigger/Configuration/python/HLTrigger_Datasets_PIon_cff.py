# getDatasets.py

import FWCore.ParameterSet.Config as cms


# dump of the Stream PhysicsHadronsTaus Datasets defined in the HLT table as Stream A Datasets

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetJetHT_selector
streamA_datasetJetHT_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetJetHT_selector.l1tResults = cms.InputTag('')
streamA_datasetJetHT_selector.throw      = cms.bool(False)
streamA_datasetJetHT_selector.triggerConditions = cms.vstring('HLT_PFJet40_v6')


# dump of the Stream PhysicsMuons Datasets defined in the HLT table as Stream A Datasets

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetSingleMuon_selector
streamA_datasetSingleMuon_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetSingleMuon_selector.l1tResults = cms.InputTag('')
streamA_datasetSingleMuon_selector.throw      = cms.bool(False)
streamA_datasetSingleMuon_selector.triggerConditions = cms.vstring('HLT_Mu50_v4')

