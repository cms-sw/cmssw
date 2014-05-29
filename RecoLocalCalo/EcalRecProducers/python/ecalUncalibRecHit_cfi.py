import FWCore.ParameterSet.Config as cms

# producer of rechits starting from digis
ecalUncalibRecHit = cms.EDProducer("EcalUncalibRecHitProducer",
#    EBdigiCollection = cms.InputTag("ecalDigis","ebDigis"),
    EBhitCollection = cms.string('EcalUncalibRecHitsEB'),
    EEdigiCollection = cms.InputTag("ecalDigis","eeDigis"),
    EEhitCollection = cms.string('EcalUncalibRecHitsEE'),
    EKdigiCollection = cms.InputTag("simEcalGlobalZeroSuppression","ekDigis"),
    EKhitCollection = cms.string('EcalUncalibRecHitsEK'),
    algo = cms.string("EcalUncalibRecHitWorkerWeights")
)
