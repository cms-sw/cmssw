import FWCore.ParameterSet.Config as cms

#for parabolic magnetic field
from Configuration.ProcessModifiers.trackingParabolicMf_cff import trackingParabolicMf

#to resolve the refToPSet_
from TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff import CkfBaseTrajectoryFilter_block

CkfTrajectoryBuilder = cms.PSet(
    ComponentType = cms.string('CkfTrajectoryBuilder'),
    propagatorAlong = cms.string('PropagatorWithMaterial'),
#    propagatorAlong = cms.string('PropagatorWithMaterialParabolicMf'),
    trajectoryFilter = cms.PSet(refToPSet_ = cms.string('CkfBaseTrajectoryFilter_block')),
    maxCand = cms.int32(5),
    intermediateCleaning = cms.bool(True),
    estimator = cms.string('Chi2'),
    TTRHBuilder = cms.string('WithTrackAngle'),
    updator = cms.string('KFUpdator'),
    alwaysUseInvalidHits = cms.bool(True),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
#    propagatorOpposite = cms.string('PropagatorWithMaterialParabolicMfOpposite'),
    lostHitPenalty = cms.double(30.0),
    #SharedSeedCheck = cms.bool(False),
    seedAs5DHit  = cms.bool(False)
)

CkfTrajectoryBuilderIterativeDefault = CkfTrajectoryBuilder.clone()
trackingParabolicMf.toModify(CkfTrajectoryBuilderIterativeDefault,
                             propagatorAlong    = 'PropagatorWithMaterialParabolicMf',
                             propagatorOpposite = 'PropagatorWithMaterialParabolicMfOpposite')
