import FWCore.ParameterSet.Config as cms

process = cms.Process("FOURTH")

process.load("FWCore.Framework.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:testEventHistory_3.root')
)

# Why does the filter module (from step 3) pass 56 events, when I
# give it a rate of 55%?
process.historytest = cms.EDAnalyzer("HistoryAnalyzer",
    expectedCount = cms.int32(56),
    historySize = cms.int32(4)
)

process.p = cms.Path(process.historytest)
