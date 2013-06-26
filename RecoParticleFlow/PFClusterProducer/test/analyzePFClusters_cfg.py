import FWCore.ParameterSet.Config as cms

process = cms.Process("ANALYSIS")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:clustering.root')
)

process.pfClusterAnalyzer = cms.EDAnalyzer("PFClusterAnalyzer",
    PFClusters = cms.InputTag("particleFlowClusterECAL"),
    verbose = cms.untracked.bool(True),
    printBlocks = cms.untracked.bool(False)
)

process.p = cms.Path(process.pfClusterAnalyzer)


