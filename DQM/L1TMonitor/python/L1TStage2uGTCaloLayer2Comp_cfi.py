import FWCore.ParameterSet.Config as cms

l1tStage2uGTCaloLayer2Comp = cms.EDAnalyzer(
    "L1TStage2uGTCaloLayer2Comp",
    calol2JetCollection    = cms.InputTag("caloStage2Digis", "Jet"),
    uGTJetCollection       = cms.InputTag("gtStage2Digis", "Jet"),
    calol2EGammaCollection = cms.InputTag("caloStage2Digis", "EGamma"),
    uGTEGammaCollection    = cms.InputTag("gtStage2Digis", "EGamma"),
    calol2TauCollection    = cms.InputTag("caloStage2Digis", "Tau"),
    uGTTauCollection       = cms.InputTag("gtStage2Digis", "Tau"),
    calol2EtSumCollection  = cms.InputTag("caloStage2Digis", "EtSum"),
    uGTEtSumCollection     = cms.InputTag("gtStage2Digis", "EtSum"),
    monitorDir = cms.untracked.string("L1T/L1TStage2uGT/calol2ouput_vs_uGTinput")
)
