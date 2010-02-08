import FWCore.ParameterSet.Config as cms

#
hlTrigReport = cms.EDAnalyzer("HLTrigReport",
    HLTriggerResults = cms.InputTag("TriggerResults","","HLT")
)

