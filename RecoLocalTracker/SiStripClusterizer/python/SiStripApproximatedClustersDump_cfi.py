import FWCore.ParameterSet.Config as cms

SiStripApproximatedClustersDump = cms.EDAnalyzer("SiStripApproximatedClustersDump",
    approximatedClustersTag = cms.InputTag("SiStripClusters2ApproxClustersv3")
)
