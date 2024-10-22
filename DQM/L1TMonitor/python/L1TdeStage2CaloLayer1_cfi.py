import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1tdeStage2CaloLayer1 = DQMEDAnalyzer('L1TdeStage2CaloLayer1',
    dataSource = cms.InputTag("caloStage2Digis", "CaloTower"),
    emulSource = cms.InputTag("valCaloStage2Layer1Digis"),
    fedRawDataLabel  = cms.InputTag("rawDataCollector"),
    histFolder = cms.string('L1TEMU/L1TdeStage2CaloLayer1'),
)
