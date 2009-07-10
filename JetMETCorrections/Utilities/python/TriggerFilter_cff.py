import FWCore.ParameterSet.Config as cms
TriggerFilter = cms.EDFilter("TriggerFilter",
    triggerResultsTag  = cms.InputTag('TriggerResults','','HLT'),
    triggerProcessName = cms.string('HLT'),
    triggerName     = cms.string('HLT_Jet30')
)
