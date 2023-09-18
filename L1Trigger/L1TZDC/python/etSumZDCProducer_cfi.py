import FWCore.ParameterSet.Config as cms

process.etSumZdcProducer = cms.EDProducer('L1TZDCProducer',
                                          zdcDigis = cms.InputTag("simHcalTriggerPrimitiveDigis"),
                                          sampleToCenterBX = cms.int32(2),
                                          bxFirst = cms.int32(-2),
                                          bxLast = cms.int32(3)
)

