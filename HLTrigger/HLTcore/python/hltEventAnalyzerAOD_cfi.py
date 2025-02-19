import FWCore.ParameterSet.Config as cms

hltEventAnalyzerAOD = cms.EDAnalyzer("HLTEventAnalyzerAOD",
    processName = cms.string("HLT"),
    triggerName = cms.string("@"),
    triggerResults = cms.InputTag("TriggerResults","","HLT"),
    triggerEvent   = cms.InputTag("hltTriggerSummaryAOD","","HLT")
)
