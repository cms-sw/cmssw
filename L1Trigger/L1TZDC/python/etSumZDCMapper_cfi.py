import FWCore.ParameterSet.Config as cms

etSumZDCMapper = cms.EDProducer(
    "L1TZDCMapper",
    hcalTPDigis = cms.InputTag("simHcalTriggerPrimitiveDigis"),
    bxFirst = cms.int32(-2),
    bxLast = cms.int32(2)
)
