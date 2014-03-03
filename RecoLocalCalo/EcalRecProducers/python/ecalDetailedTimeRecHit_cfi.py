import FWCore.ParameterSet.Config as cms

ecalDetailedTimeRecHit = cms.EDProducer("EcalDetailedTimeRecHitProducer",
    EERecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    EBRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    EBDetailedTimeRecHitCollection = cms.string('EcalRecHitsEB'),
    EEDetailedTimeRecHitCollection = cms.string('EcalRecHitsEE')
)
