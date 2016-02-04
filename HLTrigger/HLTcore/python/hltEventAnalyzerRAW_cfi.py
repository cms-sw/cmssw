import FWCore.ParameterSet.Config as cms

hltEventAnalyzerRAW = cms.EDAnalyzer("HLTEventAnalyzerRAW",
    processName = cms.string("HLT"),
    triggerName = cms.string("@"),
    triggerResults = cms.InputTag("TriggerResults","","HLT"),
    triggerEventWithRefs = cms.InputTag("hltTriggerSummaryRAW","","HLT")
)
