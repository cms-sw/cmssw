import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1tStage2CaloLayer2Emul = DQMEDAnalyzer('L1TStage2CaloLayer2',
                stage2CaloLayer2JetSource = cms.InputTag("valCaloStage2Layer2Digis"),
                stage2CaloLayer2EGammaSource = cms.InputTag("valCaloStage2Layer2Digis"),
                stage2CaloLayer2TauSource = cms.InputTag("valCaloStage2Layer2Digis"),
                stage2CaloLayer2EtSumSource = cms.InputTag("valCaloStage2Layer2Digis"),
                monitorDir = cms.untracked.string("L1TEMU/L1TStage2CaloLayer2")
)
                                     
