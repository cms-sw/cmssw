import FWCore.ParameterSet.Config as cms

ClusterShapeTrajectoryFilter = cms.PSet(
    ComponentType = cms.string('ClusterShapeTrajectoryFilter'),
    cacheSrc = cms.InputTag("siPixelClusterShapeCache")
)
