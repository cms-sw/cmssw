import FWCore.ParameterSet.Config as cms

process = cms.Process("ANALYSIS")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:step1.root')
)

process.pfClusterAnalyzer = cms.EDAnalyzer("PFClusterAnalyzer",
    PFClusters = cms.InputTag("particleFlowClusterEKUncorrected"),
    verbose = cms.untracked.bool(True),
    printBlocks = cms.untracked.bool(False)
)

process.p = cms.Path(process.pfClusterAnalyzer)


