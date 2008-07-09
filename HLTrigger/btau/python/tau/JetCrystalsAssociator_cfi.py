import FWCore.ParameterSet.Config as cms

jetCrystalsAssociator = cms.EDFilter("JetCrystalsAssociator",
    jets = cms.InputTag("DUMMYJETS"),
    EBRecHits = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    coneSize = cms.double(0.5),
    EERecHits = cms.InputTag("ecalRecHit","EcalRecHitsEE")
)


