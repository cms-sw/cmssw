import FWCore.ParameterSet.Config as cms

l1tdeStage2CaloLayer1 = cms.EDAnalyzer("L1TdeStage2CaloLayer1",
    dataSource = cms.InputTag("caloStage2Digis", "CaloTower"),
    emulSource = cms.InputTag("valCaloStage2Layer1Digis"),
    histFolder = cms.string('L1T2016EMU/L1TdeStage2CaloLayer1'),
)
