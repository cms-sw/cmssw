import FWCore.ParameterSet.Config as cms

triggerSummaryAnalyzerAOD = cms.EDFilter("TriggerSummaryAnalyzerAOD",
    inputTag = cms.InputTag("hltTriggerSummaryAOD")
)


