import FWCore.ParameterSet.Config as cms

slimmedPatTrigger = cms.EDProducer("PATTriggerObjectStandAloneSlimmer",
    src = cms.InputTag("selectedPatTrigger"),
    packFilterLabels = cms.bool(True),
    packP4 = cms.bool(True),
)
