import FWCore.ParameterSet.Config as cms

hltIter2Phase2L3FromL1TkMuonPixelClusterCheck = cms.EDProducer("ClusterCheckerEDProducer",
    ClusterCollectionLabel = cms.InputTag("hltMeasurementTrackerEvent"),
    MaxNumberOfPixelClusters = cms.uint32(10000),
    MaxNumberOfStripClusters = cms.uint32(50000),
    PixelClusterCollectionLabel = cms.InputTag("hltSiPixelClusters"),
    cut = cms.string(''),
    doClusterCheck = cms.bool(False),
    silentClusterCheck = cms.untracked.bool(False)
)
