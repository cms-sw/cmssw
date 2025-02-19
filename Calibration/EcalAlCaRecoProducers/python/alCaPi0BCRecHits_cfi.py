import FWCore.ParameterSet.Config as cms

# AlCaPi0BCRecHits producer
alCaPi0BCRecHits = cms.EDProducer("AlCaPi0BasicClusterRecHitsProducer",
    pi0BarrelHitCollection = cms.string('pi0EcalRecHitsEB'),
    seleMinvMaxPi0 = cms.double(0.2),
    gammaCandEtaSize = cms.int32(21),
    selePtGammaOne = cms.double(1.0),
    ecalRecHitsProducer = cms.string('ecalRecHit'),
    islandBCColl = cms.string('islandBarrelBasicClusters'),
    VerbosityLevel = cms.string('ERROR'),
    gammaCandPhiSize = cms.int32(21),
    selePtGammaTwo = cms.double(1.0),
    islandBCProd = cms.string('islandBasicClusters'),
    selePtPi0 = cms.double(2.5),
    barrelHitCollection = cms.string('EcalRecHitsEB'),
    seleMinvMinPi0 = cms.double(0.0)
)


