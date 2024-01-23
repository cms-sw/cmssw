import FWCore.ParameterSet.Config as cms

etSumZdcProducer = cms.EDProducer('L1TZDCProducer',
                                  hcalTPDigis = cms.InputTag("simHcalTriggerPrimitiveDigis"),
                                  bxFirst = cms.int32(-2),
                                  bxLast = cms.int32(3)
                                  )

