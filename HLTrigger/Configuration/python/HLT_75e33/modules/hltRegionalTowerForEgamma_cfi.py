import FWCore.ParameterSet.Config as cms

hltRegionalTowerForEgamma = cms.EDProducer("EgammaHLTCaloTowerProducer",
    EMin = cms.double(0.8),
    EtMin = cms.double(0.5),
    L1IsoCand = cms.InputTag("hltGtStage2Digis","EGamma"),
    L1NonIsoCand = cms.InputTag("hltGtStage2Digis","EGamma"),
    towerCollection = cms.InputTag("hltTowerMakerForAllForEgamma"),
    useTowersInCone = cms.double(0.8)
)
