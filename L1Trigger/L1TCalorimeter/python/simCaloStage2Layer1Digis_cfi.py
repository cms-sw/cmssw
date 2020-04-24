import FWCore.ParameterSet.Config as cms

simCaloStage2Layer1Digis = cms.EDProducer(
    "L1TStage2Layer1Producer",
    verbosity     = cms.int32(2),
    rctConditions = cms.bool(False),
    bxFirst       = cms.int32(0),
    bxLast        = cms.int32(0),
    ecalToken     = cms.InputTag("simEcalTriggerPrimitiveDigis"),
    hcalToken     = cms.InputTag("simHcalTriggerPrimitiveDigis"),
    firmware      = cms.int32(1)
)
