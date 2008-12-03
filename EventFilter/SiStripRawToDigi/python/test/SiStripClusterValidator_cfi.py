import FWCore.ParameterSet.Config as cms

ValidateSiStripClusters = cms.EDFilter("SiStripClusterValidator",
    Collection2 = cms.untracked.InputTag("siStripClustersDSV"),
    Collection1 = cms.untracked.InputTag("siStripClusters")
)


