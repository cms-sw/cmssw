import FWCore.ParameterSet.Config as cms

# the uGT DQM module
from DQM.L1TMonitor.L1TStage2uGT_cfi import *

# compares the unpacked uGMT muon collection to the unpacked uGT muon collection
# only muons that do not match are filled in the histograms
l1tStage2uGMTOutVsuGTIn = cms.EDAnalyzer(
    "L1TStage2MuonComp",
    muonCollection1 = cms.InputTag("gmtStage2Digis", "Muon"),
    muonCollection2 = cms.InputTag("gtStage2Digis", "Muon"),
    monitorDir = cms.untracked.string("L1T/L1TStage2uGT/uGMToutput_vs_uGTinput"),
    muonCollection1Title = cms.untracked.string("uGMT output muons"),
    muonCollection2Title = cms.untracked.string("uGT input muons"),
    summaryTitle = cms.untracked.string("Summary of comparison between uGMT output muons and uGT input muons"),
    verbose = cms.untracked.bool(False),
)

# sequences
l1tStage2uGTOnlineDQMSeq = cms.Sequence(
    l1tStage2uGT +
    l1tStage2uGMTOutVsuGTIn
)
