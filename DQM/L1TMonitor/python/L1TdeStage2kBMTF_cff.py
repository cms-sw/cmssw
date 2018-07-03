import FWCore.ParameterSet.Config as cms

# the Emulator kBMTF DQM module
from DQM.L1TMonitor.L1TdeStage2BMTF_cfi import *

# compares the unpacked kBMTF regional muon collection to the emulated kBMTF regional muon collection
#New Kalman Plots for BMTF
l1tdeStage2KalmanBmtf = l1tdeStage2Bmtf.clone()
l1tdeStage2KalmanBmtf.regionalMuonCollection1 = cms.InputTag("bmtfDigis","kBMTF")
l1tdeStage2KalmanBmtf.regionalMuonCollection2 = cms.InputTag("valKBmtfDigis", "BMTF")
l1tdeStage2KalmanBmtf.monitorDir = cms.untracked.string("L1TEMU/L1TdeStage2BMTF/L1TdeStage2KalmanBMTF")
l1tdeStage2KalmanBmtf.regionalMuonCollection1Title = cms.untracked.string("kBMTF data")
l1tdeStage2KalmanBmtf.regionalMuonCollection2Title = cms.untracked.string("kBMTF emulator")
l1tdeStage2KalmanBmtf.summaryTitle = cms.untracked.string("Summary of comparison between kBMTF muons and kBMTF emulator muons")
l1tdeStage2KalmanBmtf.ignoreBin = cms.untracked.vint32(ignoreBinsDeStage2Bmtf)
l1tdeStage2KalmanBmtf.verbose = cms.untracked.bool(False)
l1tdeStage2KalmanBmtf.kalman = cms.untracked.bool(True)



# sequences
