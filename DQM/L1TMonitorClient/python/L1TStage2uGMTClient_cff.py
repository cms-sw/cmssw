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

l1tStage2uGMTMuonVsuGMTMuonCopy1RatioClient = l1tStage2uGMTOutVsuGTInRatioClient.clone()
l1tStage2uGMTMuonVsuGMTMuonCopy1RatioClient.monitorDir = cms.untracked.string('L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy1')
l1tStage2uGMTMuonVsuGMTMuonCopy1RatioClient.inputNum = cms.untracked.string('L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy1/errorSummaryNum')
l1tStage2uGMTMuonVsuGMTMuonCopy1RatioClient.inputDen = cms.untracked.string('L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy1/errorSummaryDen')
l1tStage2uGMTMuonVsuGMTMuonCopy1RatioClient.ratioTitle = cms.untracked.string('Summary of mismatch rates between uGMT muons and uGMT muon copy 1')

l1tStage2uGMTMuonVsuGMTMuonCopy2RatioClient = l1tStage2uGMTOutVsuGTInRatioClient.clone()
l1tStage2uGMTMuonVsuGMTMuonCopy2RatioClient.monitorDir = cms.untracked.string('L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy2')
l1tStage2uGMTMuonVsuGMTMuonCopy2RatioClient.inputNum = cms.untracked.string('L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy2/errorSummaryNum')
l1tStage2uGMTMuonVsuGMTMuonCopy2RatioClient.inputDen = cms.untracked.string('L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy2/errorSummaryDen')
l1tStage2uGMTMuonVsuGMTMuonCopy2RatioClient.ratioTitle = cms.untracked.string('Summary of mismatch rates between uGMT muons and uGMT muon copy 2')

l1tStage2uGMTMuonVsuGMTMuonCopy3RatioClient = l1tStage2uGMTOutVsuGTInRatioClient.clone()
l1tStage2uGMTMuonVsuGMTMuonCopy3RatioClient.monitorDir = cms.untracked.string('L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy3')
l1tStage2uGMTMuonVsuGMTMuonCopy3RatioClient.inputNum = cms.untracked.string('L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy3/errorSummaryNum')
l1tStage2uGMTMuonVsuGMTMuonCopy3RatioClient.inputDen = cms.untracked.string('L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy3/errorSummaryDen')
l1tStage2uGMTMuonVsuGMTMuonCopy3RatioClient.ratioTitle = cms.untracked.string('Summary of mismatch rates between uGMT muons and uGMT muon copy 3')

l1tStage2uGMTMuonVsuGMTMuonCopy4RatioClient = l1tStage2uGMTOutVsuGTInRatioClient.clone()
l1tStage2uGMTMuonVsuGMTMuonCopy4RatioClient.monitorDir = cms.untracked.string('L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy4')
l1tStage2uGMTMuonVsuGMTMuonCopy4RatioClient.inputNum = cms.untracked.string('L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy4/errorSummaryNum')
l1tStage2uGMTMuonVsuGMTMuonCopy4RatioClient.inputDen = cms.untracked.string('L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy4/errorSummaryDen')
l1tStage2uGMTMuonVsuGMTMuonCopy4RatioClient.ratioTitle = cms.untracked.string('Summary of mismatch rates between uGMT muons and uGMT muon copy 4')

l1tStage2uGMTMuonVsuGMTMuonCopy5RatioClient = l1tStage2uGMTOutVsuGTInRatioClient.clone()
l1tStage2uGMTMuonVsuGMTMuonCopy5RatioClient.monitorDir = cms.untracked.string('L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy5')
l1tStage2uGMTMuonVsuGMTMuonCopy5RatioClient.inputNum = cms.untracked.string('L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy5/errorSummaryNum')
l1tStage2uGMTMuonVsuGMTMuonCopy5RatioClient.inputDen = cms.untracked.string('L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy5/errorSummaryDen')
l1tStage2uGMTMuonVsuGMTMuonCopy5RatioClient.ratioTitle = cms.untracked.string('Summary of mismatch rates between uGMT muons and uGMT muon copy 5')

# RegionalMuonCands
l1tStage2BmtfOutVsuGMTInRatioClient = l1tStage2uGMTOutVsuGTInRatioClient.clone()
l1tStage2BmtfOutVsuGMTInRatioClient.monitorDir = cms.untracked.string('L1T/L1TStage2uGMT/BMTFoutput_vs_uGMTinput')
l1tStage2BmtfOutVsuGMTInRatioClient.inputNum = cms.untracked.string('L1T/L1TStage2uGMT/BMTFoutput_vs_uGMTinput/errorSummaryNum')
l1tStage2BmtfOutVsuGMTInRatioClient.inputDen = cms.untracked.string('L1T/L1TStage2uGMT/BMTFoutput_vs_uGMTinput/errorSummaryDen')
l1tStage2BmtfOutVsuGMTInRatioClient.ratioTitle = cms.untracked.string('Summary of mismatch rates between BMTF output muons and uGMT input muons from BMTF')

l1tStage2EmtfOutVsuGMTInRatioClient = l1tStage2uGMTOutVsuGTInRatioClient.clone()
l1tStage2EmtfOutVsuGMTInRatioClient.monitorDir = cms.untracked.string('L1T/L1TStage2uGMT/EMTFoutput_vs_uGMTinput')
l1tStage2EmtfOutVsuGMTInRatioClient.inputNum = cms.untracked.string('L1T/L1TStage2uGMT/EMTFoutput_vs_uGMTinput/errorSummaryNum')
l1tStage2EmtfOutVsuGMTInRatioClient.inputDen = cms.untracked.string('L1T/L1TStage2uGMT/EMTFoutput_vs_uGMTinput/errorSummaryDen')
l1tStage2EmtfOutVsuGMTInRatioClient.ratioTitle = cms.untracked.string('Summary of mismatch rates between BMTF output muons and uGMT input muons from EMTF')

# zero suppression
l1tStage2uGMTZeroSuppRatioClient = l1tStage2uGMTOutVsuGTInRatioClient.clone()
l1tStage2uGMTZeroSuppRatioClient.monitorDir = cms.untracked.string('L1T/L1TStage2uGMT/zeroSuppression/AllEvts')
l1tStage2uGMTZeroSuppRatioClient.inputNum = cms.untracked.string('L1T/L1TStage2uGMT/zeroSuppression/AllEvts/errorSummaryNum')
l1tStage2uGMTZeroSuppRatioClient.inputDen = cms.untracked.string('L1T/L1TStage2uGMT/zeroSuppression/AllEvts/errorSummaryDen')
l1tStage2uGMTZeroSuppRatioClient.ratioTitle = cms.untracked.string('Summary of bad zero suppression rates')
l1tStage2uGMTZeroSuppRatioClient.yAxisTitle = cms.untracked.string('# fail / # total')

l1tStage2uGMTZeroSuppFatEvtsRatioClient = l1tStage2uGMTZeroSuppRatioClient.clone()
l1tStage2uGMTZeroSuppFatEvtsRatioClient.monitorDir = cms.untracked.string('L1T/L1TStage2uGMT/zeroSuppression/FatEvts')
l1tStage2uGMTZeroSuppFatEvtsRatioClient.inputNum = cms.untracked.string('L1T/L1TStage2uGMT/zeroSuppression/FatEvts/errorSummaryNum')
l1tStage2uGMTZeroSuppFatEvtsRatioClient.inputDen = cms.untracked.string('L1T/L1TStage2uGMT/zeroSuppression/FatEvts/errorSummaryDen')
l1tStage2uGMTZeroSuppFatEvtsRatioClient.ratioTitle = cms.untracked.string('Summary of bad zero suppression rates')

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

