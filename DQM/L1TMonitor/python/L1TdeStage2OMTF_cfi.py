import FWCore.ParameterSet.Config as cms
# List of bins to ignore
ignoreBinsDeStage2Omtf = [1, 14]

# compares the unpacked OMTF regional muon collection to the emulated OMTF regional muon collection
# only muons that do not match are filled in the histograms
l1tdeStage2Omtf = cms.EDAnalyzer(
    "L1TStage2RegionalMuonCandComp",
    regionalMuonCollection1 = cms.InputTag("omtfStage2Digis", ""),
    regionalMuonCollection2 = cms.InputTag("valOmtfDigis", "OMTF"),
    monitorDir = cms.untracked.string("L1TEMU/L1TdeStage2OMTF"),
    regionalMuonCollection1Title = cms.untracked.string("OMTF data"),
    regionalMuonCollection2Title = cms.untracked.string("OMTF emulator"),
    summaryTitle = cms.untracked.string("Summary of comparison between OMTF muons and OMTF emulator muons"),
    ignoreBadTrackAddress = cms.untracked.bool(True),
    ignoreBin = cms.untracked.vint32(ignoreBinsDeStage2Omtf),
    verbose = cms.untracked.bool(False),
)

