import FWCore.ParameterSet.Config as cms

# RegionalMuonCands
l1tStage2EMTFEmulatorCompRatioClient = cms.EDAnalyzer("L1TStage2RatioClient",
    monitorDir = cms.untracked.string('L1TEMU/L1TdeStage2EMTF'),
    inputNum = cms.untracked.string('L1TEMU/L1TdeStage2EMTF/errorSummaryNum'),
    inputDen = cms.untracked.string('L1TEMU/L1TdeStage2EMTF/errorSummaryDen'),
    ratioName = cms.untracked.string('mismatchRatio'),
    ratioTitle = cms.untracked.string('Summary of mismatch rates between EMTF muons and EMTF emulator muons'),
    yAxisTitle = cms.untracked.string('# mismatch / # total'),
    binomialErr = cms.untracked.bool(True)
)

# sequences
l1tStage2EMTFEmulatorClient = cms.Sequence(
    l1tStage2EMTFEmulatorCompRatioClient
)

