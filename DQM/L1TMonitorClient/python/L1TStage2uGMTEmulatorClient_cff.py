import FWCore.ParameterSet.Config as cms

# Muons
l1tStage2uGMTEmulatorCompRatioClient = cms.EDAnalyzer("L1TStage2RatioClient",
    monitorDir = cms.untracked.string('L1TEMU/L1TdeStage2uGMT/data_vs_emulator_comparison'),
    inputNum = cms.untracked.string('L1TEMU/L1TdeStage2uGMT/data_vs_emulator_comparison/errorSummaryNum'),
    inputDen = cms.untracked.string('L1TEMU/L1TdeStage2uGMT/data_vs_emulator_comparison/errorSummaryDen'),
    ratioName = cms.untracked.string('mismatchRatio'),
    ratioTitle = cms.untracked.string('Summary of mismatch rates between uGMT muons and uGMT emulator muons'),
    yAxisTitle = cms.untracked.string('# mismatch / # total'),
    binomialErr = cms.untracked.bool(True)
)

# sequences
l1tStage2uGMTEmulatorClient = cms.Sequence(
    l1tStage2uGMTEmulatorCompRatioClient
)

