import FWCore.ParameterSet.Config as cms

# re-calibrated rechit producer
ecalRecHit = cms.EDProducer("EcalRecalibRecHitProducer",
    doEnergyScale = cms.bool(False),
    doEnergyScaleInverse = cms.bool(False),
    doIntercalib = cms.bool(False),
    doIntercalibInverse = cms.bool(False),
    EERecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    EBRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    doLaserCorrections = cms.bool(False),
    doLaserCorrectionsInverse = cms.bool(False),
    EBRecalibRecHitCollection = cms.string('EcalRecHitsEB'),
    EERecalibRecHitCollection = cms.string('EcalRecHitsEE')
)
