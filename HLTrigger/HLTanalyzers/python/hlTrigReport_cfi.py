import FWCore.ParameterSet.Config as cms

#
hlTrigReport = cms.EDFilter("HLTrigReport",
    HLTriggerResults = cms.InputTag("TriggerResults","","HLT")
)


