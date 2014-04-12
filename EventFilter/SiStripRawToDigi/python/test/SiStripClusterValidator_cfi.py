import FWCore.ParameterSet.Config as cms

ValidateSiStripClusters = cms.EDFilter(
    "SiStripClusterValidator",
    Collection1 = cms.untracked.InputTag("siStripClusters"),
    Collection2 = cms.untracked.InputTag("siStripClustersDSV"),
    DetSetVectorNew = cms.untracked.bool(True),
    )

