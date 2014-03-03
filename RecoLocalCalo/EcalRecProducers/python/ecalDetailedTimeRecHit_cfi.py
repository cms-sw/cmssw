import FWCore.ParameterSet.Config as cms

ecalDetailedTimeRecHit = cms.EDProducer("EcalDetailedTimeRecHitProducer",
                                        EERecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
                                        EBRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
                                        EETimeDigiCollection = cms.InputTag("mix","EETimeDigi"),
                                        EBTimeDigiCollection = cms.InputTag("mix","EBTimeDigi"),
                                        EBDetailedTimeRecHitCollection = cms.string('EcalRecHitsEB'),
                                        EEDetailedTimeRecHitCollection = cms.string('EcalRecHitsEE')
)
