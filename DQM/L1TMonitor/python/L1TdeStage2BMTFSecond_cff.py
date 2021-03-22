import FWCore.ParameterSet.Config as cms

# the Emulator kBMTF DQM module
from DQM.L1TMonitor.L1TdeStage2BMTF_cfi import *

# compares the unpacked BMTF2 regional muon collection to the emulated BMTF2 regional muon collection (after the TriggerAlgoSelector decide which is BMTF2)
# Plots for BMTF
l1tdeStage2BmtfSecond = l1tdeStage2Bmtf.clone()
l1tdeStage2BmtfSecond.regionalMuonCollection1 = cms.InputTag("bmtfDigis","BMTF2")
l1tdeStage2BmtfSecond.regionalMuonCollection2 = cms.InputTag("valBmtfAlgoSel", "BMTF2")
l1tdeStage2BmtfSecond.monitorDir = cms.untracked.string("L1TEMU/L1TdeStage2BMTF/L1TdeStage2BMTF-Secondary")
l1tdeStage2BmtfSecond.regionalMuonCollection1Title = cms.untracked.string("BMTF2 data")
l1tdeStage2BmtfSecond.regionalMuonCollection2Title = cms.untracked.string("BMTF2 emulator")
l1tdeStage2BmtfSecond.summaryTitle = cms.untracked.string("Summary of comparison between BMTF2 muons and BMTF2 emulator muons")
l1tdeStage2BmtfSecond.ignoreBin = cms.untracked.vint32(ignoreBinsDeStage2Bmtf)
l1tdeStage2BmtfSecond.verbose = cms.untracked.bool(False)
l1tdeStage2BmtfSecond.hasDisplacementInfo = cms.untracked.bool(True)



# sequences
