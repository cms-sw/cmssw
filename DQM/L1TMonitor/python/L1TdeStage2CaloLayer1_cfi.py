import FWCore.ParameterSet.Config as cms

l1tdeStage2CaloLayer1 = DQMStep1Module('L1TdeStage2CaloLayer1',
    dataSource = cms.InputTag("caloStage2Digis", "CaloTower"),
    emulSource = cms.InputTag("valCaloStage2Layer1Digis"),
    histFolder = cms.string('L1TEMU/L1TdeStage2CaloLayer1'),
)
