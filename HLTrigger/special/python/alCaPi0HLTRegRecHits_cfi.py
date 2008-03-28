import FWCore.ParameterSet.Config as cms

# AlCaPi0RecHits HLT filter
alCaPi0RegRecHits = cms.EDFilter("HLTPi0RecHitsFilter",
    pi0BarrelHitCollection = cms.string('pi0EcalRecHitsEB'),
    seleNRHMax = cms.int32(75),
    seleMinvMaxPi0 = cms.double(0.16),
    gammaCandPhiSize = cms.int32(21),
    clusPhiSize = cms.int32(3),
    gammaCandEtaSize = cms.int32(21),
    clusEtaSize = cms.int32(3),
    #    string ecalRecHitsProducer = "ecalRegionalEgammaRecHit"
    #    string barrelHitCollection = "EcalRecHitsEB"
    #   replace the 2 strings with 1 InputTag of form label:instance
    barrelHits = cms.InputTag("ecalRegionalEgammaRecHit","EcalRecHitsEB"),
    seleMinvMinPi0 = cms.double(0.09),
    selePtGammaTwo = cms.double(1.0),
    selePtPi0 = cms.double(2.5),
    seleXtalMinEnergy = cms.double(0.0),
    selePtGammaOne = cms.double(1.0),
    clusSeedThr = cms.double(0.5)
)


