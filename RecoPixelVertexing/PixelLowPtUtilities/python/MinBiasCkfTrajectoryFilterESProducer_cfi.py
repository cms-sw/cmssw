import FWCore.ParameterSet.Config as cms

# Trajectory filter for min bias
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi
ckfBaseTrajectoryFilterForMinBias = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone(ComponentName = 'ckfBaseTrajectoryFilterForMinBias')

ckfBaseTrajectoryFilterForMinBias.filterPset.minimumNumberOfHits = 3
ckfBaseTrajectoryFilterForMinBias.filterPset.minPt               = 0.075

from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeTrajectoryFilterESProducer_cfi import *

# Composite filter
import TrackingTools.TrajectoryFiltering.CompositeTrajectoryFilterESProducer_cfi
minBiasTrajectoryFilterESProducer = TrackingTools.TrajectoryFiltering.CompositeTrajectoryFilterESProducer_cfi.compositeTrajectoryFilterESProducer.clone(
    ComponentName = cms.string('MinBiasCkfTrajectoryFilter'),
    filterNames   = cms.vstring('ckfBaseTrajectoryFilterForMinBias',
                                'clusterShapeTrajectoryFilter')
    )

