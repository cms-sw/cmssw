import FWCore.ParameterSet.Config as cms

reducedHcalRecHits = cms.EDProducer("HcalHitSelection",
                                    hbheTag = cms.InputTag('hbhereco'),
                                    hfTag = cms.InputTag('hfreco'),
                                    hoTag = cms.InputTag('horeco'),
                                    hoSeverityLevel = cms.int32(1),
                                    interestingDetIds = cms.VInputTag()
                                    )
