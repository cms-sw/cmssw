import FWCore.ParameterSet.Config as cms

l1comparatorResultDigis = cms.EDProducer(
    "l1t::L1ComparatorRun2",
    JetData = cms.InputTag("caloStage2Digis","Jet"),
    JetEmul = cms.InputTag("simCaloStage2Digis"),
    EGammaData = cms.InputTag("caloStage2Digis","EGamma"),
    EGammaEmul = cms.InputTag("simCaloStage2Digis"),
    TauData = cms.InputTag("caloStage2Digis","Tau"),
    TauEmul = cms.InputTag("simCaloStage2Digis"),
    EtSumData = cms.InputTag("caloStage2Digis","EtSum"),
    EtSumEmul = cms.InputTag("simCaloStage2Digis"),
    CaloTowerData = cms.InputTag("simCaloStage2Layer1Digis","CaloTower"),
    CaloTowerEmul = cms.InputTag("simCaloStage2Layer1Digis"),
    bxMax = cms.int32(0),
    bxMin = cms.int32(0),
    doLayer2 = cms.bool(True),
    doLayer1 = cms.bool(True)
    )

