import FWCore.ParameterSet.Config as cms

triggerSummaryAnalyzerRAW = cms.EDFilter("TriggerSummaryAnalyzerRAW",
    inputTag = cms.InputTag("hltTriggerSummaryRAW")
)


