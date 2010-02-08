import FWCore.ParameterSet.Config as cms

triggerSummaryAnalyzerAOD = cms.EDAnalyzer("TriggerSummaryAnalyzerAOD",
    inputTag = cms.InputTag("hltTriggerSummaryAOD")
)


