import FWCore.ParameterSet.Config as cms

l1tStage2CaloLayer2Emul = cms.EDAnalyzer("L1TStage2CaloLayer2",
                stage2CaloLayer2JetSource = cms.InputTag("simCaloStage2Digis"),
                stage2CaloLayer2EGammaSource = cms.InputTag("simCaloStage2Digis"),
                stage2CaloLayer2TauSource = cms.InputTag("simCaloStage2Digis"),
                stage2CaloLayer2EtSumSource = cms.InputTag("simCaloStage2Digis"),
                monitorDir = cms.untracked.string("L1T2016EMU/L1TStage2CaloLayer2EMU")
)
                                     
