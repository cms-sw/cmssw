import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TCaloLayer1.simCaloStage2Layer1Digis_cfi import simCaloStage2Layer1Digis
valCaloStage2Layer1Digis = simCaloStage2Layer1Digis.clone()
valCaloStage2Layer1Digis.ecalToken = cms.InputTag("l1tCaloLayer1Digis")
valCaloStage2Layer1Digis.hcalToken = cms.InputTag("l1tCaloLayer1Digis")

l1tdeStage2CaloLayer1 = cms.EDAnalyzer("L1TdeStage2CaloLayer1",
    dataSource = cms.InputTag("caloStage2Digis", "CaloTower"),
    emulSource = cms.InputTag("valCaloStage2Layer1Digis"),
    histFolder = cms.string('L1T2016/L1TdeStage2CaloLayer1'),
)

l1tLayer1ValSequence = cms.Sequence(
    valCaloStage2Layer1Digis +
    l1tdeStage2CaloLayer1
    )
