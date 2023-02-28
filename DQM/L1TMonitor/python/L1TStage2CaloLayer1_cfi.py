import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1tStage2CaloLayer1 = DQMEDAnalyzer('L1TStage2CaloLayer1',
    ecalTPSourceRecd = cms.InputTag("caloLayer1Digis"),
    ecalTPSourceRecdBx1 = cms.InputTag("caloLayer1Digis","EcalDigisBx1"),
    ecalTPSourceRecdBx2 = cms.InputTag("caloLayer1Digis","EcalDigisBx2"),
    ecalTPSourceRecdBx3 = cms.InputTag("caloLayer1Digis","EcalDigisBx3"),
    ecalTPSourceRecdBx4 = cms.InputTag("caloLayer1Digis","EcalDigisBx4"),
    ecalTPSourceRecdBx5 = cms.InputTag("caloLayer1Digis","EcalDigisBx5"),
    hcalTPSourceRecd = cms.InputTag("caloLayer1Digis"),
    ecalTPSourceSent = cms.InputTag("ecalDigis","EcalTriggerPrimitives"),
    hcalTPSourceSent = cms.InputTag("hcalDigis"),
    CaloTowerCollectionData  = cms.InputTag("caloStage2Digis","CaloTower"),
    fedRawDataLabel  = cms.InputTag("rawDataCollector"),
    histFolder = cms.string('L1T/L1TStage2CaloLayer1'),
    ignoreHFfb2 = cms.untracked.bool(False),
)
