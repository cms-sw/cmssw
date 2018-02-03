import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1tStage2CaloLayer2 = DQMEDAnalyzer('L1TStage2CaloLayer2',
                stage2CaloLayer2JetSource = cms.InputTag("caloStage2Digis","Jet"),
                stage2CaloLayer2EGammaSource = cms.InputTag("caloStage2Digis","EGamma"),
                stage2CaloLayer2TauSource = cms.InputTag("caloStage2Digis","Tau"),
                stage2CaloLayer2EtSumSource = cms.InputTag("caloStage2Digis","EtSum"),
                monitorDir = cms.untracked.string("L1T/L1TStage2CaloLayer2")
)
                                     
