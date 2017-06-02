import FWCore.ParameterSet.Config as cms

# input modules
unpackerModule = "gmtStage2Digis"
emulatorModule = "valGmtStage2Digis"

# directories
ugmtEmuDqmDir = "L1TEMU/L1TdeStage2uGMT"
ugmtEmuImdMuDqmDir = ugmtEmuDqmDir+"/intermediate_muons"

# fills histograms with all uGMT emulated muons
# uGMT input muon histograms from track finders are not filled since they are identical to the data DQM plots
from DQM.L1TMonitor.L1TStage2uGMT_cfi import *
l1tStage2uGMTEmul = l1tStage2uGMT.clone()
l1tStage2uGMTEmul.muonProducer = cms.InputTag(emulatorModule)
l1tStage2uGMTEmul.monitorDir = cms.untracked.string(ugmtEmuDqmDir)
l1tStage2uGMTEmul.emulator = cms.untracked.bool(True)

# the uGMT intermediate muon DQM modules
l1tStage2uGMTIntermediateBMTFEmul = cms.EDAnalyzer(
    "L1TStage2uGMTMuon",
    muonProducer = cms.InputTag(emulatorModule, "imdMuonsBMTF"),
    monitorDir = cms.untracked.string(ugmtEmuImdMuDqmDir+"/BMTF"),
    titlePrefix = cms.untracked.string("uGMT intermediate muon from BMTF "),
    verbose = cms.untracked.bool(False),
)

l1tStage2uGMTIntermediateOMTFNegEmul = cms.EDAnalyzer(
    "L1TStage2uGMTMuon",
    muonProducer = cms.InputTag(emulatorModule, "imdMuonsOMTFNeg"),
    monitorDir = cms.untracked.string(ugmtEmuImdMuDqmDir+"/OMTF_neg"),
    titlePrefix = cms.untracked.string("uGMT intermediate muon from OMTF neg. "),
    verbose = cms.untracked.bool(False),
)

l1tStage2uGMTIntermediateOMTFPosEmul = cms.EDAnalyzer(
    "L1TStage2uGMTMuon",
    muonProducer = cms.InputTag(emulatorModule, "imdMuonsOMTFPos"),
    monitorDir = cms.untracked.string(ugmtEmuImdMuDqmDir+"/OMTF_pos"),
    titlePrefix = cms.untracked.string("uGMT intermediate muon from OMTF pos. "),
    verbose = cms.untracked.bool(False),
)

l1tStage2uGMTIntermediateEMTFNegEmul = cms.EDAnalyzer(
    "L1TStage2uGMTMuon",
    muonProducer = cms.InputTag(emulatorModule, "imdMuonsEMTFNeg"),
    monitorDir = cms.untracked.string(ugmtEmuImdMuDqmDir+"/EMTF_neg"),
    titlePrefix = cms.untracked.string("uGMT intermediate muon from EMTF neg. "),
    verbose = cms.untracked.bool(False),
)

l1tStage2uGMTIntermediateEMTFPosEmul = cms.EDAnalyzer(
    "L1TStage2uGMTMuon",
    muonProducer = cms.InputTag(emulatorModule, "imdMuonsEMTFPos"),
    monitorDir = cms.untracked.string(ugmtEmuImdMuDqmDir+"/EMTF_pos"),
    titlePrefix = cms.untracked.string("uGMT intermediate muon from EMTF pos. "),
    verbose = cms.untracked.bool(False),
)

# compares the unpacked uGMT muon collection to the emulated uGMT muon collection
# only muons that do not match are filled in the histograms
l1tdeStage2uGMT = cms.EDAnalyzer(
    "L1TStage2MuonComp",
    muonCollection1 = cms.InputTag(unpackerModule, "Muon"),
    muonCollection2 = cms.InputTag(emulatorModule),
    monitorDir = cms.untracked.string(ugmtEmuDqmDir+"/data_vs_emulator_comparison"),
    muonCollection1Title = cms.untracked.string("uGMT data"),
    muonCollection2Title = cms.untracked.string("uGMT emulator"),
    summaryTitle = cms.untracked.string("Summary of comparison between uGMT muons and uGMT emulator muons"),
    verbose = cms.untracked.bool(False),
)

# compares the unpacked uGMT intermediate muon collection to the emulated uGMT intermediate muon collection
# only muons that do not match are filled in the histograms
l1tdeStage2uGMTIntermediateBMTF = l1tdeStage2uGMT.clone()
l1tdeStage2uGMTIntermediateBMTF.muonCollection1 = cms.InputTag(unpackerModule, "imdMuonsBMTF")
l1tdeStage2uGMTIntermediateBMTF.muonCollection2 = cms.InputTag(emulatorModule, "imdMuonsBMTF")
l1tdeStage2uGMTIntermediateBMTF.monitorDir = cms.untracked.string(ugmtEmuImdMuDqmDir+"/BMTF/data_vs_emulator_comparison")
l1tdeStage2uGMTIntermediateBMTF.summaryTitle = cms.untracked.string("Summary of uGMT intermediate muon from BMTF comparison between unpacked and emulated")

l1tdeStage2uGMTIntermediateOMTFNeg = l1tdeStage2uGMTIntermediateBMTF.clone()
l1tdeStage2uGMTIntermediateOMTFNeg.muonCollection1 = cms.InputTag(unpackerModule, "imdMuonsOMTFNeg")
l1tdeStage2uGMTIntermediateOMTFNeg.muonCollection2 = cms.InputTag(emulatorModule, "imdMuonsOMTFNeg")
l1tdeStage2uGMTIntermediateOMTFNeg.monitorDir = cms.untracked.string(ugmtEmuImdMuDqmDir+"/OMTF_neg/data_vs_emulator_comparison")
l1tdeStage2uGMTIntermediateOMTFNeg.summaryTitle = cms.untracked.string("Summary of uGMT intermediate muon from OMTF- comparison between unpacked and emulated")

l1tdeStage2uGMTIntermediateOMTFPos = l1tdeStage2uGMTIntermediateBMTF.clone()
l1tdeStage2uGMTIntermediateOMTFPos.muonCollection1 = cms.InputTag(unpackerModule, "imdMuonsOMTFPos")
l1tdeStage2uGMTIntermediateOMTFPos.muonCollection2 = cms.InputTag(emulatorModule, "imdMuonsOMTFPos")
l1tdeStage2uGMTIntermediateOMTFPos.monitorDir = cms.untracked.string(ugmtEmuImdMuDqmDir+"/OMTF_pos/data_vs_emulator_comparison")
l1tdeStage2uGMTIntermediateOMTFPos.summaryTitle = cms.untracked.string("Summary of uGMT intermediate muon from OMTF+ comparison between unpacked and emulated")

l1tdeStage2uGMTIntermediateEMTFNeg = l1tdeStage2uGMTIntermediateBMTF.clone()
l1tdeStage2uGMTIntermediateEMTFNeg.muonCollection1 = cms.InputTag(unpackerModule, "imdMuonsEMTFNeg")
l1tdeStage2uGMTIntermediateEMTFNeg.muonCollection2 = cms.InputTag(emulatorModule, "imdMuonsEMTFNeg")
l1tdeStage2uGMTIntermediateEMTFNeg.monitorDir = cms.untracked.string(ugmtEmuImdMuDqmDir+"/EMTF_neg/data_vs_emulator_comparison")
l1tdeStage2uGMTIntermediateEMTFNeg.summaryTitle = cms.untracked.string("Summary of uGMT intermediate muon from EMTF- comparison between unpacked and emulated")

l1tdeStage2uGMTIntermediateEMTFPos = l1tdeStage2uGMTIntermediateBMTF.clone()
l1tdeStage2uGMTIntermediateEMTFPos.muonCollection1 = cms.InputTag(unpackerModule, "imdMuonsEMTFPos")
l1tdeStage2uGMTIntermediateEMTFPos.muonCollection2 = cms.InputTag(emulatorModule, "imdMuonsEMTFPos")
l1tdeStage2uGMTIntermediateEMTFPos.monitorDir = cms.untracked.string(ugmtEmuImdMuDqmDir+"/EMTF_pos/data_vs_emulator_comparison")
l1tdeStage2uGMTIntermediateEMTFPos.summaryTitle = cms.untracked.string("Summary of uGMT intermediate muon from EMTF+ comparison between unpacked and emulated")

# sequences
l1tStage2uGMTEmulatorOnlineDQMSeq = cms.Sequence(
    l1tStage2uGMTEmul +
    l1tStage2uGMTIntermediateBMTFEmul +
    l1tStage2uGMTIntermediateOMTFNegEmul +
    l1tStage2uGMTIntermediateOMTFPosEmul +
    l1tStage2uGMTIntermediateEMTFNegEmul +
    l1tStage2uGMTIntermediateEMTFPosEmul +
    l1tdeStage2uGMT +
    l1tdeStage2uGMTIntermediateBMTF +
    l1tdeStage2uGMTIntermediateOMTFNeg +
    l1tdeStage2uGMTIntermediateOMTFPos +
    l1tdeStage2uGMTIntermediateEMTFNeg +
    l1tdeStage2uGMTIntermediateEMTFPos
)
