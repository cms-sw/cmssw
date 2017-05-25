import FWCore.ParameterSet.Config as cms

# fills histograms with all uGMT emulated muons
# uGMT input muon histograms from track finders are not filled since they are identical to the data DQM plots
from DQM.L1TMonitor.L1TStage2uGMT_cfi import *
l1tStage2uGMTEmul = l1tStage2uGMT.clone()
l1tStage2uGMTEmul.muonProducer = cms.InputTag("valGmtStage2Digis")
l1tStage2uGMTEmul.monitorDir = cms.untracked.string("L1TEMU/L1TdeStage2uGMT")
l1tStage2uGMTEmul.emulator = cms.untracked.bool(True)

# the uGMT intermediate muon DQM modules
l1tStage2uGMTIntermediateBMTFEmul = cms.EDAnalyzer(
    "L1TStage2uGMTMuon",
    muonProducer = cms.InputTag("valGmtStage2Digis", "imdMuonsBMTF"),
    monitorDir = cms.untracked.string("L1TEMU/L1TdeStage2uGMT/intermediate_muons/BMTF"),
    titlePrefix = cms.untracked.string("uGMT intermediate muon from BMTF "),
    verbose = cms.untracked.bool(False),
)

l1tStage2uGMTIntermediateOMTFNegEmul = cms.EDAnalyzer(
    "L1TStage2uGMTMuon",
    muonProducer = cms.InputTag("valGmtStage2Digis", "imdMuonsOMTFNeg"),
    monitorDir = cms.untracked.string("L1TEMU/L1TdeStage2uGMT/intermediate_muons/OMTF_neg"),
    titlePrefix = cms.untracked.string("uGMT intermediate muon from OMTF neg. "),
    verbose = cms.untracked.bool(False),
)

l1tStage2uGMTIntermediateOMTFPosEmul = cms.EDAnalyzer(
    "L1TStage2uGMTMuon",
    muonProducer = cms.InputTag("valGmtStage2Digis", "imdMuonsOMTFPos"),
    monitorDir = cms.untracked.string("L1TEMU/L1TdeStage2uGMT/intermediate_muons/OMTF_pos"),
    titlePrefix = cms.untracked.string("uGMT intermediate muon from OMTF pos. "),
    verbose = cms.untracked.bool(False),
)

l1tStage2uGMTIntermediateEMTFNegEmul = cms.EDAnalyzer(
    "L1TStage2uGMTMuon",
    muonProducer = cms.InputTag("valGmtStage2Digis", "imdMuonsEMTFNeg"),
    monitorDir = cms.untracked.string("L1TEMU/L1TdeStage2uGMT/intermediate_muons/EMTF_neg"),
    titlePrefix = cms.untracked.string("uGMT intermediate muon from EMTF neg. "),
    verbose = cms.untracked.bool(False),
)

l1tStage2uGMTIntermediateEMTFPosEmul = cms.EDAnalyzer(
    "L1TStage2uGMTMuon",
    muonProducer = cms.InputTag("valGmtStage2Digis", "imdMuonsEMTFPos"),
    monitorDir = cms.untracked.string("L1TEMU/L1TdeStage2uGMT/intermediate_muons/EMTF_pos"),
    titlePrefix = cms.untracked.string("uGMT intermediate muon from EMTF pos. "),
    verbose = cms.untracked.bool(False),
)

# compares the unpacked uGMT muon collection to the emulated uGMT muon collection
# only muons that do not match are filled in the histograms
l1tdeStage2uGMT = cms.EDAnalyzer(
    "L1TStage2MuonComp",
    muonCollection1 = cms.InputTag("gmtStage2Digis", "Muon"),
    muonCollection2 = cms.InputTag("valGmtStage2Digis"),
    monitorDir = cms.untracked.string("L1TEMU/L1TdeStage2uGMT/data_vs_emulator_comparison"),
    muonCollection1Title = cms.untracked.string("uGMT data"),
    muonCollection2Title = cms.untracked.string("uGMT emulator"),
    summaryTitle = cms.untracked.string("Summary of comparison between uGMT muons and uGMT emulator muons"),
    verbose = cms.untracked.bool(False),
)

# sequences
l1tStage2uGMTEmulatorOnlineDQMSeq = cms.Sequence(
    l1tStage2uGMTEmul +
    l1tStage2uGMTIntermediateBMTFEmul +
    l1tStage2uGMTIntermediateOMTFNegEmul +
    l1tStage2uGMTIntermediateOMTFPosEmul +
    l1tStage2uGMTIntermediateEMTFNegEmul +
    l1tStage2uGMTIntermediateEMTFPosEmul +
    l1tdeStage2uGMT
)
