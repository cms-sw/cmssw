import FWCore.ParameterSet.Config as cms

# rechit producer
ecal2004TBRecHit = cms.EDProducer("EcalRecHitProducer",
    EErechitCollection = cms.string(''),
    EEuncalibRecHitCollection = cms.InputTag(""),
    EBuncalibRecHitCollection = cms.InputTag("ecal2004TBWeightUncalibRecHit","EcalUncalibRecHitsEB"),
    EBrechitCollection = cms.string("EcalRecHitsEB"),
    algo = cms.string("EcalRecHitWorkerSimple")
)
