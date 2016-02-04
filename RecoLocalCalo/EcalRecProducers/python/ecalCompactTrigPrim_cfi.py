import FWCore.ParameterSet.Config as cms

ecalCompactTrigPrim = cms.EDProducer("EcalCompactTrigPrimProducer",
                                     #inColl = cms.InputTag("simEcalTriggerPrimitiveDigis:"),
                                     inColl = cms.InputTag("ecalDigis:EcalTriggerPrimitives"),
                                     outColl = cms.string("")
                                     )
