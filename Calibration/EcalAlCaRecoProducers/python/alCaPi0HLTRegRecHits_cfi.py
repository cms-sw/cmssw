import FWCore.ParameterSet.Config as cms

# AlCaPi0RecHits producer
alCaPi0RegRecHits = cms.EDProducer("AlCaPi0RecHitsProducer",
    pi0BarrelHitCollection = cms.string('pi0EcalRecHitsEB'),
    seleNRHMax = cms.int32(75),
    clusPhiSize = cms.int32(3),
    seleMinvMaxPi0 = cms.double(0.16),
    gammaCandEtaSize = cms.int32(21),
    selePtGammaOne = cms.double(1.0),
    ecalRecHitsProducer = cms.string('ecalRegionalEgammaRecHit'),
    clusEtaSize = cms.int32(3),
    VerbosityLevel = cms.string('ERROR'),
    gammaCandPhiSize = cms.int32(21),
    selePtGammaTwo = cms.double(1.0),
    selePtPi0 = cms.double(2.5),
    seleXtalMinEnergy = cms.double(0.0),
    barrelHitCollection = cms.string('EcalRecHitsEB'),
    seleMinvMinPi0 = cms.double(0.09),
    clusSeedThr = cms.double(0.5)
)


