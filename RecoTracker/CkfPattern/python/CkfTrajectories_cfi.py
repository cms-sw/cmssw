import FWCore.ParameterSet.Config as cms

# for parabolic magnetic field
from Configuration.ProcessModifiers.trackingParabolicMf_cff import trackingParabolicMf

#
# This cfi should be included to run the CkfTrajectoryMaker 
#
ckfTrajectories = cms.EDProducer("CkfTrajectoryMaker",
    # these two needed by HLT
    cleanTrajectoryAfterInOut = cms.bool( False ),
    maxNSeeds = cms.uint32( 100000 ),
    # set it as "none" to avoid redundant seed cleaner
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    #string RedundantSeedCleaner  = "none"
    useHitsSplitting = cms.bool(False),
    doSeedingRegionRebuilding = cms.bool(False),
    ## reverse trajectories after pattern-reco creating new seed on last hit
    reverseTrajectories       = cms.bool(False),
    trackCandidateAlso = cms.bool(False),
    #bool   seedCleaning         = false
    src = cms.InputTag('globalMixedSeeds'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryBuilder = cms.string('GroupedCkfTrajectoryBuilder'),
    TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string('GroupedCkfTrajectoryBuilder')),
    # nested parameter set for TransientInitialStateEstimator
    TransientInitialStateEstimatorParameters = cms.PSet(
       propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
       propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite'),
#       propagatorAlongTISE = cms.string('PropagatorWithMaterialParabolicMf'),
#       propagatorOppositeTISE = cms.string('PropagatorWithMaterialParabolicMfOpposite'),
       numberMeasurementsForFit = cms.int32(4)
    ),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent")
)

ckfTrajectoriesIterativeDefault = ckfTrajectories.clone(
    TrajectoryBuilder = cms.string('GroupedCkfTrajectoryBuilderIterativeDefault'),
    TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string('GroupedCkfTrajectoryBuilderIterativeDefault')),
)
trackingParabolicMf.toModify(ckfTrajectoriesIterativeDefault,
                             NavigationSchool='SimpleNavigationSchoolParabolicMf')
trackingParabolicMf.toModify(ckfTrajectoriesIterativeDefault.TransientInitialStateEstimatorParameters,
                             propagatorAlongTISE   ='PropagatorWithMaterialParabolicMf',
                             propagatorOppositeTISE='PropagatorWithMaterialParabolicMfOpposite')
