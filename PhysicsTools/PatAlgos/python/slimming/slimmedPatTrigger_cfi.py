import FWCore.ParameterSet.Config as cms

slimmedPatTrigger = cms.EDProducer("PATTriggerObjectStandAloneSlimmer",
    src = cms.InputTag("selectedPatTrigger"),
    triggerResults              = cms.InputTag( 'TriggerResults::HLT' ),
    packFilterLabels = cms.bool(True),
    packP4 = cms.bool(True),
)
