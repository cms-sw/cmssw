import FWCore.ParameterSet.Config as cms

from TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cff import *
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi
ckfBaseTrajectoryFilterForMinBias = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone()
clusterShapeTrajectoryFilterESProducer = cms.ESProducer("ClusterShapeTrajectoryFilterESProducer",
    filterPset = cms.PSet(
        ComponentType = cms.string('clusterShapeTrajectoryFilter')
    ),
    ComponentName = cms.string('clusterShapeTrajectoryFilter')
)

minBiasTrajectoryFilterESProducer = cms.ESProducer("CompositeTrajectoryFilterESProducer",
    ComponentName = cms.string('MinBiasCkfTrajectoryFilter'),
    filterNames = cms.vstring('ckfBaseTrajectoryFilterForMinBias', 
        'clusterShapeTrajectoryFilter')
)

ckfBaseTrajectoryFilterForMinBias.ComponentName = 'ckfBaseTrajectoryFilterForMinBias'
ckfBaseTrajectoryFilterForMinBias.filterPset.minimumNumberOfHits = 3
ckfBaseTrajectoryFilterForMinBias.filterPset.minPt = 0.075

