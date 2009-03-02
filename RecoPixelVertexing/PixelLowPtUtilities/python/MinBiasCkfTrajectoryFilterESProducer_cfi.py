import FWCore.ParameterSet.Config as cms

from TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cff import *

# Trajectory filter for min bias
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi
ckfBaseTrajectoryFilterForMinBias = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone()
ckfBaseTrajectoryFilterForMinBias.ComponentName = 'ckfBaseTrajectoryFilterForMinBias'
ckfBaseTrajectoryFilterForMinBias.filterPset.minimumNumberOfHits = 3
ckfBaseTrajectoryFilterForMinBias.filterPset.minPt               = 0.075

# Cluster shape filter
clusterShapeTrajectoryFilterESProducer = cms.ESProducer("ClusterShapeTrajectoryFilterESProducer",
    filterPset = cms.PSet(
        ComponentType = cms.string('clusterShapeTrajectoryFilter')
    ),
    ComponentName = cms.string('clusterShapeTrajectoryFilter')
)

# Composite filter
minBiasTrajectoryFilterESProducer = cms.ESProducer("CompositeTrajectoryFilterESProducer",
    ComponentName = cms.string('MinBiasCkfTrajectoryFilter'),
    filterNames   = cms.vstring('ckfBaseTrajectoryFilterForMinBias',
                                'clusterShapeTrajectoryFilter')
)

