import FWCore.ParameterSet.Config as cms

# fills histograms with all uGMT emulated muons
# uGMT input muon histograms from track finders are not filled since they are identical to the data DQM plots
l1tStage2uGMTEmul = cms.EDAnalyzer(
    "L1TStage2uGMT",
    bmtfProducer = cms.InputTag("gmtStage2Digis", "BMTF"), # not used for emulator DQM
    omtfProducer = cms.InputTag("gmtStage2Digis", "OMTF"), # not used for emulator DQM
    emtfProducer = cms.InputTag("gmtStage2Digis", "EMTF"), # not used for emulator DQM
    muonProducer = cms.InputTag("valGmtStage2Digis"),
    monitorDir = cms.untracked.string("L1T2016EMU/L1TdeStage2uGMT"),
    emulator = cms.untracked.bool(True),
    verbose = cms.untracked.bool(False),
)

# compares the unpacked uGMT muon collection to the emulated uGMT muon collection
# only muons that do not match are filled in the histograms
l1tdeStage2uGMT = cms.EDAnalyzer(
    "L1TStage2MuonComp",
    muonCollection1 = cms.InputTag("gmtStage2Digis", "Muon"),
    muonCollection2 = cms.InputTag("valGmtStage2Digis"),
    monitorDir = cms.untracked.string("L1T2016EMU/L1TdeStage2uGMT/data_vs_emulator_comparison"),
    muonCollection1Title = cms.untracked.string("uGMT data"),
    muonCollection2Title = cms.untracked.string("uGMT emulator"),
    summaryTitle = cms.untracked.string("Summary of comparison between uGMT muons and uGMT emulator muons"),
    verbose = cms.untracked.bool(False),
)

