import FWCore.ParameterSet.Config as cms

tifClusterFilter = cms.EDFilter("ClusterMultiplicityFilter",
    MaxNumberOfClusters = cms.untracked.uint32(300),
    ClusterCollectionLabel = cms.untracked.string('siStripClusters')
)


