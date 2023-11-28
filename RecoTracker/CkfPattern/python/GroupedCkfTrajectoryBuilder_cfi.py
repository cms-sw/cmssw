import FWCore.ParameterSet.Config as cms

# for parabolic magnetic field
from Configuration.ProcessModifiers.trackingParabolicMf_cff import trackingParabolicMf

# to resolve the refToPSet_
from TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff import CkfBaseTrajectoryFilter_block

GroupedCkfTrajectoryBuilder = cms.PSet(
    ComponentType = cms.string('GroupedCkfTrajectoryBuilder'),
    bestHitOnly = cms.bool(True),
    propagatorAlong = cms.string('PropagatorWithMaterial'),
#    propagatorAlong = cms.string('PropagatorWithMaterialParabolicMf'),
    # Filter used on tracks at end of all tracking (in-out + out-in)
    trajectoryFilter = cms.PSet(refToPSet_ = cms.string('CkfBaseTrajectoryFilter_block')),
    # Filter used on tracks at end of in-out tracking phase
    inOutTrajectoryFilter = cms.PSet(refToPSet_ = cms.string('CkfBaseTrajectoryFilter_block')),
#    inOutTrajectoryFilter = cms.PSet(refToPSet_ = cms.string('ckfBaseInOutTrajectoryFilter')),
    # If true, then the inOutTrajectoryFilter will be ignored
    # and the trajectoryFilter will be used for in-out tracking too.
    useSameTrajFilter = cms.bool(True),
    # Maximum number of track candidates followed at each step of
    # track building
    maxCand = cms.int32(5),
    intermediateCleaning = cms.bool(True),
    # Chi2 added to track candidate if no hit found in layer
    lostHitPenalty = cms.double(30.0),
    foundHitBonus = cms.double(10.0),
    lockHits = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    updator = cms.string('KFUpdator'),
    # If true, track building will allow for possibility of no hit
    # in a given layer, even if it finds compatible hits there.
    alwaysUseInvalidHits = cms.bool(True),
    requireSeedHitsInRebuild = cms.bool(True),
    keepOriginalIfRebuildFails = cms.bool(False),
    estimator = cms.string('Chi2'),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
#    propagatorOpposite = cms.string('PropagatorWithMaterialParabolicMfOpposite'),
    # Out-in tracking will not be attempted unless this many hits
    # are on track after in-out tracking phase.
    minNrOfHitsForRebuild = cms.int32(5),
    seedAs5DHit = cms.bool(False),
    maxPtForLooperReconstruction = cms.double(0.),
    maxDPhiForLooperReconstruction = cms.double(2.),
)

GroupedCkfTrajectoryBuilderIterativeDefault = GroupedCkfTrajectoryBuilder.clone()
trackingParabolicMf.toModify(GroupedCkfTrajectoryBuilderIterativeDefault,
                             propagatorAlong='PropagatorWithMaterialParabolicMf',
                             propagatorOpposite='PropagatorWithMaterialParabolicMfOpposite')
