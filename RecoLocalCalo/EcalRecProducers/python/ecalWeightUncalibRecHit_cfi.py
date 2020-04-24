import FWCore.ParameterSet.Config as cms

# producer of rechits starting from digis
ecalWeightUncalibRecHit = cms.EDProducer("EcalUncalibRecHitProducer",
    EBdigiCollection = cms.InputTag("ecalDigis","ebDigis"),
    EEhitCollection = cms.string('EcalUncalibRecHitsEE'),
    EEdigiCollection = cms.InputTag("ecalDigis","eeDigis"),
    EBhitCollection = cms.string('EcalUncalibRecHitsEB'),
    algo = cms.string("EcalUncalibRecHitWorkerWeights"),
    algoPSet = cms.PSet()
)
