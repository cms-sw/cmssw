import FWCore.ParameterSet.Config as cms

ecalMaxSampleUncalibRecHit = cms.EDProducer("EcalUncalibRecHitProducer",
    EBdigiCollection = cms.InputTag("ecalEBunpacker","ebDigis"),
    EEhitCollection = cms.string("EcalUncalibRecHitsEE"),
    EEdigiCollection = cms.InputTag("ecalEBunpacker","eeDigis"),
    EBhitCollection = cms.string("EcalUncalibRecHitsEB"),
    algo = cms.string("EcalUncalibRecHitWorkerMaxSample"),
    algoPSet = cms.PSet()
)
