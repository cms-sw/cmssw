import FWCore.ParameterSet.Config as cms

# compares the unpacked uGMT input regional muon collection from OMTF to the emulated OMTF regional muon collection
# only muons that do not match are filled in the histograms
l1tdeStage2Omtf = cms.EDAnalyzer(
    "L1TStage2RegionalMuonCandComp",
    regionalMuonCollection1 = cms.InputTag("gmtStage2Digis", "OMTF"),
    regionalMuonCollection2 = cms.InputTag("valOmtfDigis", "OMTF"),
    monitorDir = cms.untracked.string("L1T2016EMU/L1TdeStage2OMTF"),
    regionalMuonCollection1Title = cms.untracked.string("uGMT input data from OMTF"),
    regionalMuonCollection2Title = cms.untracked.string("OMTF emulator"),
    summaryTitle = cms.untracked.string("Summary of comparison between uGMT input muons from OMTF and OMTF emulator muons"),
    ignoreBadTrackAddress = cms.untracked.bool(True),
    verbose = cms.untracked.bool(False),
)

