import FWCore.ParameterSet.Config as cms

# directory path shortening
ugmtDqmDir = 'L1T/L1TStage2uGMT'
ugmtMuCpyDqmDir = ugmtDqmDir+'/uGMTMuonCopies'
ugmtZSDqmDir = ugmtDqmDir+'/zeroSuppression'
# input histograms
errHistNumStr = 'errorSummaryNum'
errHistDenStr = 'errorSummaryDen'

# Muons
l1tStage2uGMTOutVsuGTInRatioClient = cms.EDAnalyzer("L1TStage2RatioClient",
    monitorDir = cms.untracked.string(ugmtDqmDir+'/uGMToutput_vs_uGTinput'),
    inputNum = cms.untracked.string(ugmtDqmDir+'/uGMToutput_vs_uGTinput/'+errHistNumStr),
    inputDen = cms.untracked.string(ugmtDqmDir+'/uGMToutput_vs_uGTinput/'+errHistDenStr),
    ratioName = cms.untracked.string('mismatchRatio'),
    ratioTitle = cms.untracked.string('Summary of mismatch rates between uGMT output muons and uGT input muons'),
    yAxisTitle = cms.untracked.string('# mismatch / # total'),
    binomialErr = cms.untracked.bool(True)
)

l1tStage2uGMTMuonVsuGMTMuonCopy1RatioClient = l1tStage2uGMTOutVsuGTInRatioClient.clone()
l1tStage2uGMTMuonVsuGMTMuonCopy1RatioClient.monitorDir = cms.untracked.string(ugmtMuCpyDqmDir+'/uGMTMuonCopy1')
l1tStage2uGMTMuonVsuGMTMuonCopy1RatioClient.inputNum = cms.untracked.string(ugmtMuCpyDqmDir+'/uGMTMuonCopy1/'+errHistNumStr)
l1tStage2uGMTMuonVsuGMTMuonCopy1RatioClient.inputDen = cms.untracked.string(ugmtMuCpyDqmDir+'/uGMTMuonCopy1/'+errHistDenStr)
l1tStage2uGMTMuonVsuGMTMuonCopy1RatioClient.ratioTitle = cms.untracked.string('Summary of mismatch rates between uGMT muons and uGMT muon copy 1')

l1tStage2uGMTMuonVsuGMTMuonCopy2RatioClient = l1tStage2uGMTOutVsuGTInRatioClient.clone()
l1tStage2uGMTMuonVsuGMTMuonCopy2RatioClient.monitorDir = cms.untracked.string(ugmtMuCpyDqmDir+'/uGMTMuonCopy2')
l1tStage2uGMTMuonVsuGMTMuonCopy2RatioClient.inputNum = cms.untracked.string(ugmtMuCpyDqmDir+'/uGMTMuonCopy2/'+errHistNumStr)
l1tStage2uGMTMuonVsuGMTMuonCopy2RatioClient.inputDen = cms.untracked.string(ugmtMuCpyDqmDir+'/uGMTMuonCopy2/'+errHistDenStr)
l1tStage2uGMTMuonVsuGMTMuonCopy2RatioClient.ratioTitle = cms.untracked.string('Summary of mismatch rates between uGMT muons and uGMT muon copy 2')

l1tStage2uGMTMuonVsuGMTMuonCopy3RatioClient = l1tStage2uGMTOutVsuGTInRatioClient.clone()
l1tStage2uGMTMuonVsuGMTMuonCopy3RatioClient.monitorDir = cms.untracked.string(ugmtMuCpyDqmDir+'/uGMTMuonCopy3')
l1tStage2uGMTMuonVsuGMTMuonCopy3RatioClient.inputNum = cms.untracked.string(ugmtMuCpyDqmDir+'/uGMTMuonCopy3/'+errHistNumStr)
l1tStage2uGMTMuonVsuGMTMuonCopy3RatioClient.inputDen = cms.untracked.string(ugmtMuCpyDqmDir+'/uGMTMuonCopy3/'+errHistDenStr)
l1tStage2uGMTMuonVsuGMTMuonCopy3RatioClient.ratioTitle = cms.untracked.string('Summary of mismatch rates between uGMT muons and uGMT muon copy 3')

l1tStage2uGMTMuonVsuGMTMuonCopy4RatioClient = l1tStage2uGMTOutVsuGTInRatioClient.clone()
l1tStage2uGMTMuonVsuGMTMuonCopy4RatioClient.monitorDir = cms.untracked.string(ugmtMuCpyDqmDir+'/uGMTMuonCopy4')
l1tStage2uGMTMuonVsuGMTMuonCopy4RatioClient.inputNum = cms.untracked.string(ugmtMuCpyDqmDir+'/uGMTMuonCopy4/'+errHistNumStr)
l1tStage2uGMTMuonVsuGMTMuonCopy4RatioClient.inputDen = cms.untracked.string(ugmtMuCpyDqmDir+'/uGMTMuonCopy4/'+errHistDenStr)
l1tStage2uGMTMuonVsuGMTMuonCopy4RatioClient.ratioTitle = cms.untracked.string('Summary of mismatch rates between uGMT muons and uGMT muon copy 4')

l1tStage2uGMTMuonVsuGMTMuonCopy5RatioClient = l1tStage2uGMTOutVsuGTInRatioClient.clone()
l1tStage2uGMTMuonVsuGMTMuonCopy5RatioClient.monitorDir = cms.untracked.string(ugmtMuCpyDqmDir+'/uGMTMuonCopy5')
l1tStage2uGMTMuonVsuGMTMuonCopy5RatioClient.inputNum = cms.untracked.string(ugmtMuCpyDqmDir+'/uGMTMuonCopy5/'+errHistNumStr)
l1tStage2uGMTMuonVsuGMTMuonCopy5RatioClient.inputDen = cms.untracked.string(ugmtMuCpyDqmDir+'/uGMTMuonCopy5/'+errHistDenStr)
l1tStage2uGMTMuonVsuGMTMuonCopy5RatioClient.ratioTitle = cms.untracked.string('Summary of mismatch rates between uGMT muons and uGMT muon copy 5')

# RegionalMuonCands
l1tStage2BmtfOutVsuGMTInRatioClient = l1tStage2uGMTOutVsuGTInRatioClient.clone()
l1tStage2BmtfOutVsuGMTInRatioClient.monitorDir = cms.untracked.string(ugmtDqmDir+'/BMTFoutput_vs_uGMTinput')
l1tStage2BmtfOutVsuGMTInRatioClient.inputNum = cms.untracked.string(ugmtDqmDir+'/BMTFoutput_vs_uGMTinput/'+errHistNumStr)
l1tStage2BmtfOutVsuGMTInRatioClient.inputDen = cms.untracked.string(ugmtDqmDir+'/BMTFoutput_vs_uGMTinput/'+errHistDenStr)
l1tStage2BmtfOutVsuGMTInRatioClient.ratioTitle = cms.untracked.string('Summary of mismatch rates between BMTF output muons and uGMT input muons from BMTF')

l1tStage2EmtfOutVsuGMTInRatioClient = l1tStage2uGMTOutVsuGTInRatioClient.clone()
l1tStage2EmtfOutVsuGMTInRatioClient.monitorDir = cms.untracked.string(ugmtDqmDir+'/EMTFoutput_vs_uGMTinput')
l1tStage2EmtfOutVsuGMTInRatioClient.inputNum = cms.untracked.string(ugmtDqmDir+'/EMTFoutput_vs_uGMTinput/'+errHistNumStr)
l1tStage2EmtfOutVsuGMTInRatioClient.inputDen = cms.untracked.string(ugmtDqmDir+'/EMTFoutput_vs_uGMTinput/'+errHistDenStr)
l1tStage2EmtfOutVsuGMTInRatioClient.ratioTitle = cms.untracked.string('Summary of mismatch rates between BMTF output muons and uGMT input muons from EMTF')

# zero suppression
l1tStage2uGMTZeroSuppRatioClient = l1tStage2uGMTOutVsuGTInRatioClient.clone()
l1tStage2uGMTZeroSuppRatioClient.monitorDir = cms.untracked.string(ugmtZSDqmDir+'/AllEvts')
l1tStage2uGMTZeroSuppRatioClient.inputNum = cms.untracked.string(ugmtZSDqmDir+'/AllEvts/'+errHistNumStr)
l1tStage2uGMTZeroSuppRatioClient.inputDen = cms.untracked.string(ugmtZSDqmDir+'/AllEvts/'+errHistDenStr)
l1tStage2uGMTZeroSuppRatioClient.ratioTitle = cms.untracked.string('Summary of bad zero suppression rates')
l1tStage2uGMTZeroSuppRatioClient.yAxisTitle = cms.untracked.string('# fail / # total')

l1tStage2uGMTZeroSuppFatEvtsRatioClient = l1tStage2uGMTZeroSuppRatioClient.clone()
l1tStage2uGMTZeroSuppFatEvtsRatioClient.monitorDir = cms.untracked.string(ugmtZSDqmDir+'/FatEvts')
l1tStage2uGMTZeroSuppFatEvtsRatioClient.inputNum = cms.untracked.string(ugmtZSDqmDir+'/FatEvts/'+errHistNumStr)
l1tStage2uGMTZeroSuppFatEvtsRatioClient.inputDen = cms.untracked.string(ugmtZSDqmDir+'/FatEvts/'+errHistDenStr)
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

