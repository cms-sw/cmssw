import FWCore.ParameterSet.Config as cms

# RegionalMuonCands
l1tStage2OMTFEmulatorCompRatioClient = cms.EDAnalyzer("L1TStage2RatioClient",
    monitorDir = cms.untracked.string('L1TEMU/L1TdeStage2OMTF'),
    inputNum = cms.untracked.string('L1TEMU/L1TdeStage2OMTF/errorSummaryNum'),
    inputDen = cms.untracked.string('L1TEMU/L1TdeStage2OMTF/errorSummaryDen'),
    ratioName = cms.untracked.string('mismatchRatio'),
    ratioTitle = cms.untracked.string('Summary of mismatch rates between OMTF muons and OMTF emulator muons'),
    yAxisTitle = cms.untracked.string('# mismatch / # total'),
    binomialErr = cms.untracked.bool(True)
)

# sequences
l1tStage2OMTFEmulatorClient = cms.Sequence(
    l1tStage2OMTFEmulatorCompRatioClient
)

