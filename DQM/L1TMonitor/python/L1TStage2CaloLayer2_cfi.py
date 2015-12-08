import FWCore.ParameterSet.Config as cms

l1tStage2CaloLayer2 = cms.EDAnalyzer("L1TStage2CaloLayer2",
                stage2CaloLayer2JetSource = cms.InputTag("caloStage2Digis"),
                stage2CaloLayer2EGammaSource = cms.InputTag("caloStage2Digis"),
                stage2CaloLayer2TauSource = cms.InputTag("caloStage2Digis"),
                stage2CaloLayer2EtSumSource = cms.InputTag("caloStage2Digis"),
                monitorDir = cms.untracked.string("L1T2016/L1TStage2CaloLayer2")
)
                                     
