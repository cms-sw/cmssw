import FWCore.ParameterSet.Config as cms

l1tStage2CaloLayer2Comp = cms.EDProducer(
    'L1TStage2CaloLayer2Comp',
    dumpTowers = cms.bool(False),
    dumpWholeEvent = cms.bool(False),
    calol2JetCollectionData    = cms.InputTag("caloStage2Digis", "MP"),
    calol2JetCollectionEmul    = cms.InputTag("simCaloStage2Digis", "MP"),
    calol2EGammaCollectionData = cms.InputTag("caloStage2Digis", "MP"),
    calol2EGammaCollectionEmul = cms.InputTag("simCaloStage2Digis", "MP"),
    calol2TauCollectionData    = cms.InputTag("caloStage2Digis", "MP"),
    calol2TauCollectionEmul    = cms.InputTag("simCaloStage2Digis", "MP"),
    calol2EtSumCollectionData  = cms.InputTag("caloStage2Digis", "MP"),
    calol2EtSumCollectionEmul  = cms.InputTag("simCaloStage2Digis", "MP"),
    calol2CaloTowerCollectionData  = cms.InputTag("caloStage2Digis", "CaloTower"),
    calol2CaloTowerCollectionEmul  = cms.InputTag("simCaloStage2Digis", "MP"),
)
