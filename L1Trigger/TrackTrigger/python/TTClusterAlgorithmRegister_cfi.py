import FWCore.ParameterSet.Config as cms

# First register all the clustering algorithms, then specify preferred ones at end.

# Clustering algorithm a
TTClusterAlgorithm_a_Phase2TrackerDigi_ = cms.ESProducer("TTClusterAlgorithm_a_Phase2TrackerDigi_")

# Broadside clustering algorithm
# Set WidthCut=0 to eliminate the width cut.
TTClusterAlgorithm_broadside_Phase2TrackerDigi_ = cms.ESProducer("TTClusterAlgorithm_broadside_Phase2TrackerDigi_",
    WidthCut = cms.int32(4)
)

# 2d clustering algorithm
TTClusterAlgorithm_2d_Phase2TrackerDigi_ = cms.ESProducer("TTClusterAlgorithm_2d_Phase2TrackerDigi_",
    DoubleCountingTest=cms.bool(True)
)

# 2d2013 clustering algorithm
TTClusterAlgorithm_2d2013_Phase2TrackerDigi_ = cms.ESProducer("TTClusterAlgorithm_2d2013_Phase2TrackerDigi_",
    WidthCut = cms.int32(4)
)

# Neighbor clustering algorithm
TTClusterAlgorithm_neighbor_Phase2TrackerDigi_ = cms.ESProducer("TTClusterAlgorithm_neighbor_Phase2TrackerDigi_")

# Set the preferred hit matching algorithms.
# We prefer the a algorithm for now in order not to break anything.
# Override with process.TTClusterAlgorithm_PSimHit_ = ..., etc. in your
# configuration.
TTClusterAlgorithm_Phase2TrackerDigi_ = cms.ESPrefer("TTClusterAlgorithm_broadside_Phase2TrackerDigi_")

