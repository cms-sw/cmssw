
import FWCore.ParameterSet.Config as cms

caloTowersFilter = cms.EDFilter(
    "L1TCaloTowersFilter",
    towerToken = cms.InputTag("caloStage2Digis", "CaloTower"),
    # inputTag  = cms.InputTag("rawDataCollector"),
    # period    = cms.untracked.int32(107)
)
