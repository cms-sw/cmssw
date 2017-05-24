import FWCore.ParameterSet.Config as cms

# Muons
l1tStage2uGMTOutVsuGTInRatioClient = cms.EDAnalyzer("L1TStage2RatioClient",
    monitorDir = cms.untracked.string('L1T/L1TStage2uGMT/uGMToutput_vs_uGTinput'),
    inputNum = cms.untracked.string('L1T/L1TStage2uGMT/uGMToutput_vs_uGTinput/errorSummaryNum'),
    inputDen = cms.untracked.string('L1T/L1TStage2uGMT/uGMToutput_vs_uGTinput/errorSummaryDen'),
    ratioName = cms.untracked.string('mismatchRatio'),
    ratioTitle = cms.untracked.string('Summary of mismatch rates between uGMT output muons and uGT input muons'),
    yAxisTitle = cms.untracked.string('# mismatch / # total'),
    binomialErr = cms.untracked.bool(True)
)

l1tStage2uGMTMuonVsuGMTMuonCopy1RatioClient = cms.EDAnalyzer("L1TStage2RatioClient",
    monitorDir = cms.untracked.string('L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy1'),
    inputNum = cms.untracked.string('L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy1/errorSummaryNum'),
    inputDen = cms.untracked.string('L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy1/errorSummaryDen'),
    ratioName = cms.untracked.string('mismatchRatio'),
    ratioTitle = cms.untracked.string('Summary of mismatch rates between uGMT muons and uGMT muon copy 1'),
    yAxisTitle = cms.untracked.string('# mismatch / # total'),
    binomialErr = cms.untracked.bool(True)
)

l1tStage2uGMTMuonVsuGMTMuonCopy2RatioClient = cms.EDAnalyzer("L1TStage2RatioClient",
    monitorDir = cms.untracked.string('L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy2'),
    inputNum = cms.untracked.string('L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy2/errorSummaryNum'),
    inputDen = cms.untracked.string('L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy2/errorSummaryDen'),
    ratioName = cms.untracked.string('mismatchRatio'),
    ratioTitle = cms.untracked.string('Summary of mismatch rates between uGMT muons and uGMT muon copy 2'),
    yAxisTitle = cms.untracked.string('# mismatch / # total'),
    binomialErr = cms.untracked.bool(True)
)

l1tStage2uGMTMuonVsuGMTMuonCopy3RatioClient = cms.EDAnalyzer("L1TStage2RatioClient",
    monitorDir = cms.untracked.string('L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy3'),
    inputNum = cms.untracked.string('L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy3/errorSummaryNum'),
    inputDen = cms.untracked.string('L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy3/errorSummaryDen'),
    ratioName = cms.untracked.string('mismatchRatio'),
    ratioTitle = cms.untracked.string('Summary of mismatch rates between uGMT muons and uGMT muon copy 3'),
    yAxisTitle = cms.untracked.string('# mismatch / # total'),
    binomialErr = cms.untracked.bool(True)
)

l1tStage2uGMTMuonVsuGMTMuonCopy4RatioClient = cms.EDAnalyzer("L1TStage2RatioClient",
    monitorDir = cms.untracked.string('L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy4'),
    inputNum = cms.untracked.string('L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy4/errorSummaryNum'),
    inputDen = cms.untracked.string('L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy4/errorSummaryDen'),
    ratioName = cms.untracked.string('mismatchRatio'),
    ratioTitle = cms.untracked.string('Summary of mismatch rates between uGMT muons and uGMT muon copy 4'),
    yAxisTitle = cms.untracked.string('# mismatch / # total'),
    binomialErr = cms.untracked.bool(True)
)

l1tStage2uGMTMuonVsuGMTMuonCopy5RatioClient = cms.EDAnalyzer("L1TStage2RatioClient",
    monitorDir = cms.untracked.string('L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy5'),
    inputNum = cms.untracked.string('L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy5/errorSummaryNum'),
    inputDen = cms.untracked.string('L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy5/errorSummaryDen'),
    ratioName = cms.untracked.string('mismatchRatio'),
    ratioTitle = cms.untracked.string('Summary of mismatch rates between uGMT muons and uGMT muon copy 5'),
    yAxisTitle = cms.untracked.string('# mismatch / # total'),
    binomialErr = cms.untracked.bool(True)
)

# RegionalMuonCands
l1tStage2BmtfOutVsuGMTInRatioClient = cms.EDAnalyzer("L1TStage2RatioClient",
    monitorDir = cms.untracked.string('L1T/L1TStage2uGMT/BMTFoutput_vs_uGMTinput'),
    inputNum = cms.untracked.string('L1T/L1TStage2uGMT/BMTFoutput_vs_uGMTinput/errorSummaryNum'),
    inputDen = cms.untracked.string('L1T/L1TStage2uGMT/BMTFoutput_vs_uGMTinput/errorSummaryDen'),
    ratioName = cms.untracked.string('mismatchRatio'),
    ratioTitle = cms.untracked.string('Summary of mismatch rates between BMTF output muons and uGMT input muons from BMTF'),
    yAxisTitle = cms.untracked.string('# mismatch / # total'),
    binomialErr = cms.untracked.bool(True)
)

l1tStage2EmtfOutVsuGMTInRatioClient = cms.EDAnalyzer("L1TStage2RatioClient",
    monitorDir = cms.untracked.string('L1T/L1TStage2uGMT/EMTFoutput_vs_uGMTinput'),
    inputNum = cms.untracked.string('L1T/L1TStage2uGMT/EMTFoutput_vs_uGMTinput/errorSummaryNum'),
    inputDen = cms.untracked.string('L1T/L1TStage2uGMT/EMTFoutput_vs_uGMTinput/errorSummaryDen'),
    ratioName = cms.untracked.string('mismatchRatio'),
    ratioTitle = cms.untracked.string('Summary of mismatch rates between BMTF output muons and uGMT input muons from EMTF'),
    yAxisTitle = cms.untracked.string('# mismatch / # total'),
    binomialErr = cms.untracked.bool(True)
)

# zero suppression
l1tStage2uGMTZeroSuppRatioClient = cms.EDAnalyzer("L1TStage2RatioClient",
    monitorDir = cms.untracked.string('L1T/L1TStage2uGMT/zeroSuppression/AllEvts'),
    inputNum = cms.untracked.string('L1T/L1TStage2uGMT/zeroSuppression/AllEvts/errorSummaryNum'),
    inputDen = cms.untracked.string('L1T/L1TStage2uGMT/zeroSuppression/AllEvts/errorSummaryDen'),
    ratioName = cms.untracked.string('mismatchRatio'),
    ratioTitle = cms.untracked.string('Summary of bad zero suppression rates'),
    yAxisTitle = cms.untracked.string('# fail / # total'),
    binomialErr = cms.untracked.bool(True)
)

l1tStage2uGMTZeroSuppFatEvtsRatioClient = cms.EDAnalyzer("L1TStage2RatioClient",
    monitorDir = cms.untracked.string('L1T/L1TStage2uGMT/zeroSuppression/FatEvts'),
    inputNum = cms.untracked.string('L1T/L1TStage2uGMT/zeroSuppression/FatEvts/errorSummaryNum'),
    inputDen = cms.untracked.string('L1T/L1TStage2uGMT/zeroSuppression/FatEvts/errorSummaryDen'),
    ratioName = cms.untracked.string('mismatchRatio'),
    ratioTitle = cms.untracked.string('Summary of bad zero suppression rates'),
    yAxisTitle = cms.untracked.string('# fail / # total'),
    binomialErr = cms.untracked.bool(True)
)

# sequences
l1tStage2uGMTMuonCompClient = cms.Sequence(
    l1tStage2uGMTOutVsuGTInRatioClient
  + l1tStage2uGMTMuonVsuGMTMuonCopy1RatioClient
  + l1tStage2uGMTMuonVsuGMTMuonCopy2RatioClient
  + l1tStage2uGMTMuonVsuGMTMuonCopy3RatioClient
  + l1tStage2uGMTMuonVsuGMTMuonCopy4RatioClient
  + l1tStage2uGMTMuonVsuGMTMuonCopy5RatioClient
)

l1tStage2uGMTRegionalMuonCandCompClient = cms.Sequence(
    l1tStage2BmtfOutVsuGMTInRatioClient
  + l1tStage2EmtfOutVsuGMTInRatioClient
)

l1tStage2uGMTZeroSuppCompClient = cms.Sequence(
    l1tStage2uGMTZeroSuppRatioClient
  + l1tStage2uGMTZeroSuppFatEvtsRatioClient
)

l1tStage2uGMTClient = cms.Sequence(
    l1tStage2uGMTMuonCompClient
  + l1tStage2uGMTRegionalMuonCandCompClient
  + l1tStage2uGMTZeroSuppCompClient
)

