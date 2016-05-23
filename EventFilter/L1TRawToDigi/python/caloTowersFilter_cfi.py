
import FWCore.ParameterSet.Config as cms

caloTowersFilter = cms.EDFilter(
    "L1TCaloTowersFilter",
    towerToken = cms.InputTag("simCaloStage2Digis", "MP"),
    # inputTag  = cms.InputTag("rawDataCollector"),
    # period    = cms.untracked.int32(107)
)
