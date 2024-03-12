import FWCore.ParameterSet.Config as cms

tifClusterFilter = cms.EDFilter("ClusterMultiplicityFilter",
    MaxNumberOfClusters = cms.uint32(300),
    ClusterCollection = cms.InputTag('siStripClusters')
)


# foo bar baz
# GrGqxOWP1dvp6
