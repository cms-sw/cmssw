import FWCore.ParameterSet.Config as cms

l1tdeStage2CaloLayer2 = cms.EDAnalyzer("L1TdeStage2CaloLayer2",
    calol2JetCollectionData = cms.InputTag("caloStage2Digis", "Jet"),
    calol2JetCollectionEmul = cms.InputTag("valCaloStage2Layer2Digis"),
    calol2EGammaCollectionData = cms.InputTag("caloStage2Digis", "EGamma"),
    calol2EGammaCollectionEmul = cms.InputTag("valCaloStage2Layer2Digis"),
    calol2TauCollectionData = cms.InputTag("caloStage2Digis", "Tau"),
    calol2TauCollectionEmul = cms.InputTag("valCaloStage2Layer2Digis"),
    calol2EtSumCollectionData = cms.InputTag("caloStage2Digis", "EtSum"),
    calol2EtSumCollectionEmul = cms.InputTag("valCaloStage2Layer2Digis"),
    monitorDir = cms.untracked.string("L1TEMU/L1TdeStage2CaloLayer2")
)
