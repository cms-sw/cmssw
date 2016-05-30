import FWCore.ParameterSet.Config as cms

l1tStage2CaloLayer2Emul = cms.EDAnalyzer("L1TStage2CaloLayer2",
                stage2CaloLayer2JetSource = cms.InputTag("valCaloStage2Layer2Digis"),
                stage2CaloLayer2EGammaSource = cms.InputTag("valCaloStage2Layer2Digis"),
                stage2CaloLayer2TauSource = cms.InputTag("valCaloStage2Layer2Digis"),
                stage2CaloLayer2EtSumSource = cms.InputTag("valCaloStage2Layer2Digis"),
                monitorDir = cms.untracked.string("L1T2016EMU/L1TStage2CaloLayer2EMU")
)
                                     
