import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1tStage2uGTCaloLayer2Comp = DQMEDAnalyzer(
    "L1TStage2uGTCaloLayer2Comp",
    collection1Title = cms.untracked.string("CaloLayer2"),
    collection2Title = cms.untracked.string("uGT"),
    JetCollection1    = cms.InputTag("caloStage2Digis", "Jet"),
    JetCollection2    = cms.InputTag("gtStage2Digis",   "Jet"),
    EGammaCollection1 = cms.InputTag("caloStage2Digis", "EGamma"),
    EGammaCollection2 = cms.InputTag("gtStage2Digis",   "EGamma"),
    TauCollection1    = cms.InputTag("caloStage2Digis", "Tau"),
    TauCollection2    = cms.InputTag("gtStage2Digis",   "Tau"),
    EtSumCollection1  = cms.InputTag("caloStage2Digis", "EtSum"),
    EtSumCollection2  = cms.InputTag("gtStage2Digis",   "EtSum"),
    monitorDir = cms.untracked.string("L1T/L1TStage2uGT/calol2ouput_vs_uGTinput")
)
