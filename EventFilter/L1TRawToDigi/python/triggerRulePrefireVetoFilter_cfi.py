import FWCore.ParameterSet.Config as cms

triggerRulePrefireVetoFilter = cms.EDFilter(
    "TriggerRulePrefireVetoFilter",
    tcdsRecordLabel = cms.InputTag("tcdsDigis","tcdsRecord"),
)
