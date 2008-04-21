import FWCore.ParameterSet.Config as cms

# Base
from TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cff import *
import copy
from TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi import *
ckfBaseTrajectoryFilterForMinBias = copy.deepcopy(trajectoryFilterESProducer)
# ClusterShape
clusterShapeTrajectoryFilterESProducer = cms.ESProducer("ClusterShapeTrajectoryFilterESProducer",
    filterPset = cms.PSet(
        ComponentType = cms.string('clusterShapeTrajectoryFilter')
    ),
    ComponentName = cms.string('clusterShapeTrajectoryFilter')
)

# Composite
minBiasTrajectoryFilterESProducer = cms.ESProducer("CompositeTrajectoryFilterESProducer",
    ComponentName = cms.string('MinBiasCkfTrajectoryFilter'),
    filterNames = cms.vstring('ckfBaseTrajectoryFilterForMinBias', 
        'clusterShapeTrajectoryFilter')
)

ckfBaseTrajectoryFilterForMinBias.ComponentName = 'ckfBaseTrajectoryFilterForMinBias'
ckfBaseTrajectoryFilterForMinBias.filterPset.minimumNumberOfHits = 3
ckfBaseTrajectoryFilterForMinBias.filterPset.minPt = 0.075

