import FWCore.ParameterSet.Config as cms

# input modules
unpackerModule = "gmtStage2Digis"
emulatorModule = "valGmtStage2Digis"
showerEmulatorModule = "valGmtShowerDigis"

# directories
ugmtEmuDqmDir = "L1TEMU/L1TdeStage2uGMT"
ugmtEmuImdMuDqmDir = ugmtEmuDqmDir+"/intermediate_muons"

# List of bins to ignore
ignoreFinalsBinsRun3 = [1]
ignoreIntermediatesBins = [7, 8, 12, 13]
ignoreIntermediatesBinsRun3 = [1, 7, 8, 12, 13]

# fills histograms with all uGMT emulated muons
# uGMT input muon histograms from track finders are not filled since they are identical to the data DQM plots
from DQM.L1TMonitor.L1TStage2uGMT_cfi import *
l1tStage2uGMTEmul = l1tStage2uGMT.clone(
    muonProducer = emulatorModule,
    monitorDir = ugmtEmuDqmDir,
    emulator = True
)
# the uGMT intermediate muon DQM modules
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1tStage2uGMTIntermediateBMTFEmul = DQMEDAnalyzer(
    "L1TStage2uGMTMuon",
    muonProducer = cms.InputTag(emulatorModule, "imdMuonsBMTF"),
    monitorDir = cms.untracked.string(ugmtEmuImdMuDqmDir+"/BMTF"),
    titlePrefix = cms.untracked.string("uGMT intermediate muon from BMTF "),
    verbose = cms.untracked.bool(False),
    displacedQuantities = cms.untracked.bool(False),
)

## Era: Run3_2021; Displaced muons from BMTF used in uGMT from Run-3
from Configuration.Eras.Modifier_stage2L1Trigger_2021_cff import stage2L1Trigger_2021
stage2L1Trigger_2021.toModify(l1tStage2uGMTIntermediateBMTFEmul, displacedQuantities = cms.untracked.bool(True))

l1tStage2uGMTIntermediateOMTFNegEmul = DQMEDAnalyzer(
    "L1TStage2uGMTMuon",
    muonProducer = cms.InputTag(emulatorModule, "imdMuonsOMTFNeg"),
    monitorDir = cms.untracked.string(ugmtEmuImdMuDqmDir+"/OMTF_neg"),
    titlePrefix = cms.untracked.string("uGMT intermediate muon from OMTF neg. "),
    verbose = cms.untracked.bool(False),
)

l1tStage2uGMTIntermediateOMTFPosEmul = DQMEDAnalyzer(
    "L1TStage2uGMTMuon",
    muonProducer = cms.InputTag(emulatorModule, "imdMuonsOMTFPos"),
    monitorDir = cms.untracked.string(ugmtEmuImdMuDqmDir+"/OMTF_pos"),
    titlePrefix = cms.untracked.string("uGMT intermediate muon from OMTF pos. "),
    verbose = cms.untracked.bool(False),
)

l1tStage2uGMTIntermediateEMTFNegEmul = DQMEDAnalyzer(
    "L1TStage2uGMTMuon",
    muonProducer = cms.InputTag(emulatorModule, "imdMuonsEMTFNeg"),
    monitorDir = cms.untracked.string(ugmtEmuImdMuDqmDir+"/EMTF_neg"),
    titlePrefix = cms.untracked.string("uGMT intermediate muon from EMTF neg. "),
    verbose = cms.untracked.bool(False),
)

l1tStage2uGMTIntermediateEMTFPosEmul = DQMEDAnalyzer(
    "L1TStage2uGMTMuon",
    muonProducer = cms.InputTag(emulatorModule, "imdMuonsEMTFPos"),
    monitorDir = cms.untracked.string(ugmtEmuImdMuDqmDir+"/EMTF_pos"),
    titlePrefix = cms.untracked.string("uGMT intermediate muon from EMTF pos. "),
    verbose = cms.untracked.bool(False),
)

## Era: Run3_2021; Displaced muons from EMTF used in uGMT from Run-3
stage2L1Trigger_2021.toModify(l1tStage2uGMTIntermediateEMTFNegEmul, displacedQuantities = cms.untracked.bool(True))
stage2L1Trigger_2021.toModify(l1tStage2uGMTIntermediateEMTFPosEmul, displacedQuantities = cms.untracked.bool(True))

# compares the unpacked uGMT muon collection to the emulated uGMT muon collection
# only muons that do not match are filled in the histograms
l1tdeStage2uGMT = DQMEDAnalyzer(
    "L1TStage2MuonComp",
    muonCollection1 = cms.InputTag(unpackerModule, "Muon"),
    muonCollection2 = cms.InputTag(emulatorModule),
    monitorDir = cms.untracked.string(ugmtEmuDqmDir+"/data_vs_emulator_comparison"),
    muonCollection1Title = cms.untracked.string("uGMT data"),
    muonCollection2Title = cms.untracked.string("uGMT emulator"),
    summaryTitle = cms.untracked.string("Summary of comparison between uGMT muons and uGMT emulator muons"),
    verbose = cms.untracked.bool(False),
    enable2DComp = cms.untracked.bool(True), # When true eta-phi comparison plots are also produced
    displacedQuantities = cms.untracked.bool(False),
    ignoreBin = cms.untracked.vint32(),
)

## Era: Run3_2021; Displaced muons used in uGMT from Run-3
 # Additionally: Ignore BX range mismatches. This is necessary because we only read out the central BX for the inputs, so that is what the emulator has to work on.
from Configuration.Eras.Modifier_stage2L1Trigger_2021_cff import stage2L1Trigger_2021
stage2L1Trigger_2021.toModify(l1tdeStage2uGMT, displacedQuantities = cms.untracked.bool(True), ignoreBin = ignoreFinalsBinsRun3)

# compares the unpacked uGMT muon shower collection to the emulated uGMT muon shower collection
# only showers that do not match are filled in the histograms
l1tdeStage2uGMTShowers = DQMEDAnalyzer(
    "L1TStage2MuonShowerComp",
    muonShowerCollection1 = cms.InputTag(unpackerModule, "MuonShower"),
    muonShowerCollection2 = cms.InputTag(showerEmulatorModule),
    monitorDir = cms.untracked.string(ugmtEmuDqmDir+"/data_vs_emulator_comparison/Muon showers"),
    muonShowerCollection1Title = cms.untracked.string("uGMT data"),
    muonShowerCollection2Title = cms.untracked.string("uGMT emulator"),
    summaryTitle = cms.untracked.string("Summary of comparison between uGMT showers and uGMT emulator showers"),
    verbose = cms.untracked.bool(False),
    ignoreBin = cms.untracked.vint32(ignoreFinalsBinsRun3), # Ignore BX range mismatches. This is necessary because we only read out the central BX for the inputs, so that is what the emulator has to work on.
)

# compares the unpacked uGMT intermediate muon collection to the emulated uGMT intermediate muon collection
# only muons that do not match are filled in the histograms
l1tdeStage2uGMTIntermediateBMTF = l1tdeStage2uGMT.clone(
    muonCollection1 = (unpackerModule, "imdMuonsBMTF"),
    muonCollection2 = (emulatorModule, "imdMuonsBMTF"),
    monitorDir = ugmtEmuImdMuDqmDir+"/BMTF/data_vs_emulator_comparison",
    summaryTitle = "Summary of uGMT intermediate muon from BMTF comparison between unpacked and emulated",
    ignoreBin = ignoreIntermediatesBins
)
## Era: Run3_2021; Ignore BX range mismatches. This is necessary because we only read out the central BX for the inputs, so that is what the emulator has to work on.
from Configuration.Eras.Modifier_stage2L1Trigger_2021_cff import stage2L1Trigger_2021
stage2L1Trigger_2021.toModify(l1tdeStage2uGMTIntermediateBMTF, ignoreBin = ignoreIntermediatesBinsRun3)

l1tdeStage2uGMTIntermediateOMTFNeg = l1tdeStage2uGMTIntermediateBMTF.clone(
    displacedQuantities = False,
    muonCollection1 = (unpackerModule, "imdMuonsOMTFNeg"),
    muonCollection2 = (emulatorModule, "imdMuonsOMTFNeg"),
    monitorDir = (ugmtEmuImdMuDqmDir+"/OMTF_neg/data_vs_emulator_comparison"),
    summaryTitle = ("Summary of uGMT intermediate muon from OMTF- comparison between unpacked and emulated"),
)
l1tdeStage2uGMTIntermediateOMTFPos = l1tdeStage2uGMTIntermediateBMTF.clone(
    displacedQuantities = False,
    muonCollection1 = (unpackerModule, "imdMuonsOMTFPos"),
    muonCollection2 = (emulatorModule, "imdMuonsOMTFPos"),
    monitorDir = (ugmtEmuImdMuDqmDir+"/OMTF_pos/data_vs_emulator_comparison"),
    summaryTitle = "Summary of uGMT intermediate muon from OMTF+ comparison between unpacked and emulated",
)
l1tdeStage2uGMTIntermediateEMTFNeg = l1tdeStage2uGMTIntermediateBMTF.clone(
    displacedQuantities = False,
    muonCollection1 = (unpackerModule, "imdMuonsEMTFNeg"),
    muonCollection2 = (emulatorModule, "imdMuonsEMTFNeg"),
    monitorDir = (ugmtEmuImdMuDqmDir+"/EMTF_neg/data_vs_emulator_comparison"),
    summaryTitle = "Summary of uGMT intermediate muon from EMTF- comparison between unpacked and emulated",
)
l1tdeStage2uGMTIntermediateEMTFPos = l1tdeStage2uGMTIntermediateBMTF.clone(
    displacedQuantities = False,
    muonCollection1 = (unpackerModule, "imdMuonsEMTFPos"),
    muonCollection2 = (emulatorModule, "imdMuonsEMTFPos"),
    monitorDir = (ugmtEmuImdMuDqmDir+"/EMTF_pos/data_vs_emulator_comparison"),
    summaryTitle = "Summary of uGMT intermediate muon from EMTF+ comparison between unpacked and emulated",
)
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

_run3_l1tStage2uGMTEmulatorOnlineDQMSeq = cms.Sequence(l1tStage2uGMTEmulatorOnlineDQMSeq.copy() + l1tdeStage2uGMTShowers)
stage2L1Trigger_2021.toReplaceWith(l1tStage2uGMTEmulatorOnlineDQMSeq, _run3_l1tStage2uGMTEmulatorOnlineDQMSeq)

