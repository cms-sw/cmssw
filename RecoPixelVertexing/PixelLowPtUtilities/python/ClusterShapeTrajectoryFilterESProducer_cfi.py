import FWCore.ParameterSet.Config as cms

# Cluster shape trajectory filter
ClusterShapeTrajectoryFilterESProducer = cms.ESProducer("ClusterShapeTrajectoryFilterESProducer",
    ComponentName = cms.string('clusterShapeTrajectoryFilter')
)
