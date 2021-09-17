import FWCore.ParameterSet.Config as cms

# List of bins to ignore
ignoreBinsDeStage2Bmtf = [1]

# compares the unpacked BMTF regional muon collection to the emulated BMTF regional muon collection
# only muons that do not match are filled in the histograms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1tdeStage2Bmtf = DQMEDAnalyzer(
    "L1TStage2RegionalMuonCandComp",
    regionalMuonCollection1 = cms.InputTag("bmtfDigis", "BMTF"),
    # regionalMuonCollection2 = cms.InputTag("valBmtfDigis", "BMTF"), # didn't remove the default config
    regionalMuonCollection2 = cms.InputTag("valBmtfAlgoSel", "BMTF"),
    monitorDir = cms.untracked.string("L1TEMU/L1TdeStage2BMTF"),
    regionalMuonCollection1Title = cms.untracked.string("BMTF data"),
    regionalMuonCollection2Title = cms.untracked.string("BMTF emulator"),
    summaryTitle = cms.untracked.string("Summary of comparison between BMTF muons and BMTF emulator muons"),
    ignoreBin = cms.untracked.vint32(ignoreBinsDeStage2Bmtf),
    verbose = cms.untracked.bool(False),
    hasDisplacementInfo = cms.untracked.bool(True)
)

