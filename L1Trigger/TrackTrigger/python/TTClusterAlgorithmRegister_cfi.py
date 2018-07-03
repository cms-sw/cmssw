import FWCore.ParameterSet.Config as cms

# First register all the clustering algorithms, then specify preferred ones at end.

# official clustering algorithm
TTClusterAlgorithm_official_Phase2TrackerDigi_ = cms.ESProducer("TTClusterAlgorithm_official_Phase2TrackerDigi_",
    WidthCut = cms.int32(4)
)

# Neighbor clustering algorithm
TTClusterAlgorithm_neighbor_Phase2TrackerDigi_ = cms.ESProducer("TTClusterAlgorithm_neighbor_Phase2TrackerDigi_")

# Set the preferred hit matching algorithms.
# We prefer the a algorithm for now in order not to break anything.
# Override with process.TTClusterAlgorithm_PSimHit_ = ..., etc. in your
# configuration.
TTClusterAlgorithm_Phase2TrackerDigi_ = cms.ESPrefer("TTClusterAlgorithm_official_Phase2TrackerDigi_")

