import FWCore.ParameterSet.Config as cms

selectedPatTrigger = cms.EDFilter("PATTriggerObjectStandAloneSelector",
    src = cms.InputTag("patTrigger"),
    cut = cms.string("!filterLabels.empty()")
)
