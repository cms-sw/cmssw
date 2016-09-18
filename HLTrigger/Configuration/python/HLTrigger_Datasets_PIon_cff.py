# /dev/CMSSW_8_0_0/PIon

import FWCore.ParameterSet.Config as cms


# stream PhysicsCommissioning

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetHLTPhysics_selector
streamPhysicsCommissioning_datasetHLTPhysics_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetHLTPhysics_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetHLTPhysics_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetHLTPhysics_selector.triggerConditions = cms.vstring('HLT_Physics_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetTemplates_selector
streamPhysicsCommissioning_datasetTemplates_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetTemplates_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetTemplates_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetTemplates_selector.triggerConditions = cms.vstring('HLT_CaloJet260_v4', 
    'HLT_Photon20_CaloIdVL_IsoL_v7')


# stream PhysicsEGamma

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsEGamma_datasetDoubleEG_selector
streamPhysicsEGamma_datasetDoubleEG_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsEGamma_datasetDoubleEG_selector.l1tResults = cms.InputTag('')
streamPhysicsEGamma_datasetDoubleEG_selector.throw      = cms.bool(False)
streamPhysicsEGamma_datasetDoubleEG_selector.triggerConditions = cms.vstring('HLT_Ele17_CaloIdL_TrackIdL_IsoVL_v7')


# stream PhysicsHadronsTaus

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHadronsTaus_datasetJetHT_selector
streamPhysicsHadronsTaus_datasetJetHT_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHadronsTaus_datasetJetHT_selector.l1tResults = cms.InputTag('')
streamPhysicsHadronsTaus_datasetJetHT_selector.throw      = cms.bool(False)
streamPhysicsHadronsTaus_datasetJetHT_selector.triggerConditions = cms.vstring('HLT_PFJet40_v9')


# stream PhysicsMuons

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsMuons_datasetSingleMuon_selector
streamPhysicsMuons_datasetSingleMuon_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsMuons_datasetSingleMuon_selector.l1tResults = cms.InputTag('')
streamPhysicsMuons_datasetSingleMuon_selector.throw      = cms.bool(False)
streamPhysicsMuons_datasetSingleMuon_selector.triggerConditions = cms.vstring('HLT_Mu50_v5')

