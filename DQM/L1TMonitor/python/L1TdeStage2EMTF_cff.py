import FWCore.ParameterSet.Config as cms

from DQM.L1TMonitor.L1TdeStage2EMTF_cfi import *
from DQM.L1TMonitor.L1TdeStage2RegionalShower_cfi import *

# List of bins to ignore
ignoreBinsDeStage2Emtf = [1]

# compares the unpacked EMTF regional muon collection to the emulated EMTF regional muon collection
# only muons that do not match are filled in the histograms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1tdeStage2EmtfComp = DQMEDAnalyzer(
    "L1TStage2RegionalMuonCandComp",
    regionalMuonCollection1 = cms.InputTag("emtfStage2Digis"),
    regionalMuonCollection2 = cms.InputTag("valEmtfStage2Digis", "EMTF"),
    monitorDir = cms.untracked.string("L1TEMU/L1TdeStage2EMTF"),
    regionalMuonCollection1Title = cms.untracked.string("EMTF data"),
    regionalMuonCollection2Title = cms.untracked.string("EMTF emulator"),
    summaryTitle = cms.untracked.string("Summary of comparison between EMTF muons and EMTF emulator muons"),
    ignoreBadTrackAddress = cms.untracked.bool(True),
    ignoreBin = cms.untracked.vint32(ignoreBinsDeStage2Emtf),
    verbose = cms.untracked.bool(False),
)

# sequences
l1tdeStage2EmtfOnlineDQMSeq = cms.Sequence(
    l1tdeStage2Emtf +
    l1tdeStage2EmtfComp
)

from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
_run3shower_l1tdeStage2EmtfOnlineDQMSeq = l1tdeStage2EmtfOnlineDQMSeq.copy()
run3_GEM.toReplaceWith(l1tdeStage2EmtfOnlineDQMSeq, cms.Sequence(_run3shower_l1tdeStage2EmtfOnlineDQMSeq + l1tdeStage2RegionalShower))
