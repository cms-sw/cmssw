import FWCore.ParameterSet.Config as cms

# Cluster shape trajectory filter
ClusterShapeTrajectoryFilter = cms.PSet(
    ComponentType = cms.string('ClusterShapeTrajectoryFilter'),
    cacheSrc = cms.InputTag('siPixelClusterShapeCache'),
)
