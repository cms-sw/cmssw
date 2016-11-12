import FWCore.ParameterSet.Config as cms

# compares the unpacked BMTF regional muon collection to the emulated BMTF regional muon collection
# only muons that do not match are filled in the histograms
l1tdeStage2Bmtf = cms.EDAnalyzer(
    "L1TStage2RegionalMuonCandComp",
    regionalMuonCollection1 = cms.InputTag("bmtfDigis", "BMTF"),
    regionalMuonCollection2 = cms.InputTag("valBmtfDigis", "BMTF"),
    monitorDir = cms.untracked.string("L1T2016EMU/L1TdeStage2BMTF"),
    regionalMuonCollection1Title = cms.untracked.string("BMTF data"),
    regionalMuonCollection2Title = cms.untracked.string("BMTF emulator"),
    summaryTitle = cms.untracked.string("Summary of comparison between BMTF muons and BMTF emulator muons"),
    verbose = cms.untracked.bool(False),
)

