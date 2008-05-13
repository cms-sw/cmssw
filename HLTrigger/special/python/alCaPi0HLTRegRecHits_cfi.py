import FWCore.ParameterSet.Config as cms

# AlCaPi0RecHits HLT filter
alCaPi0RegRecHits = cms.EDFilter("HLTPi0RecHitsFilter",
    seleS4S9GammaOne = cms.double(0.85),
    seleMinvMaxPi0 = cms.double(0.22),
    gammaCandPhiSize = cms.int32(21),
    selePtGammaOne = cms.double(0.9),
    ParameterX0 = cms.double(0.89),
    seleXtalMinEnergy = cms.double(0.0),
    selePtPi0 = cms.double(2.5),
    clusSeedThr = cms.double(0.5),
    clusPhiSize = cms.int32(3),
    selePi0BeltDR = cms.double(0.2),
    clusEtaSize = cms.int32(3),
    selePi0Iso = cms.double(0.5),
    ParameterW0 = cms.double(4.2),
    seleNRHMax = cms.int32(1000),
    selePi0BeltDeta = cms.double(0.05),
    #    string ecalRecHitsProducer = "ecalRegionalEgammaRecHit"
    #    string barrelHitCollection = "EcalRecHitsEB"
    #   replace the 2 strings with 1 InputTag of form label:instance
    barrelHits = cms.InputTag("ecalRegionalEgammaPi0RecHit","EcalRecHitsPi0EB"),
    ParameterLogWeighted = cms.bool(True),
    seleS4S9GammaTwo = cms.double(0.85),
    pi0BarrelHitCollection = cms.string('pi0EcalRecHitsEB'),
    seleMinvMinPi0 = cms.double(0.06),
    gammaCandEtaSize = cms.int32(9),
    selePtGammaTwo = cms.double(0.9),
    ParameterT0_barl = cms.double(5.7)
)


