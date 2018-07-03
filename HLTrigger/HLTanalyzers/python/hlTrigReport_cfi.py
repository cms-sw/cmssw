import FWCore.ParameterSet.Config as cms

#
hlTrigReport = cms.EDAnalyzer("HLTrigReport",
    HLTriggerResults = cms.InputTag("TriggerResults","","HLT"),
    reportBy         = cms.untracked.string("job"),
    resetBy          = cms.untracked.string("never"),
    serviceBy        = cms.untracked.string("never")
)

