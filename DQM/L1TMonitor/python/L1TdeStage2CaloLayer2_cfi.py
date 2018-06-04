import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1tdeStage2CaloLayer2 = DQMEDAnalyzer('L1TdeStage2CaloLayer2',
    calol2JetCollectionData = cms.InputTag("caloStage2Digis", "Jet"),
    calol2JetCollectionEmul = cms.InputTag("valCaloStage2Layer2Digis"),
    calol2EGammaCollectionData = cms.InputTag("caloStage2Digis", "EGamma"),
    calol2EGammaCollectionEmul = cms.InputTag("valCaloStage2Layer2Digis"),
    calol2TauCollectionData = cms.InputTag("caloStage2Digis", "Tau"),
    calol2TauCollectionEmul = cms.InputTag("valCaloStage2Layer2Digis"),
    calol2EtSumCollectionData = cms.InputTag("caloStage2Digis", "EtSum"),
    calol2EtSumCollectionEmul = cms.InputTag("valCaloStage2Layer2Digis"),
    monitorDir = cms.untracked.string("L1TEMU/L1TStage2CaloLayer2/L1TdeStage2CaloLayer2"),
    enable2DComp = cms.untracked.bool(True) # When true eta-phi comparison plots are also produced
)
