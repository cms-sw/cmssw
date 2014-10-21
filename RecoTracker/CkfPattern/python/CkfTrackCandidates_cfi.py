import FWCore.ParameterSet.Config as cms

ckfTrackCandidates = cms.EDProducer("CkfTrackCandidateMaker",
# During tracking, eliminate seeds used by an already found track 
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
# Decide how to eliminate tracks sharing hits at end of tracking phase
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
# Run cleaning after in-out tracking in addition to at end of tracking ?
    cleanTrajectoryAfterInOut = cms.bool(True),
# Split matched strip tracker hits into mono/stereo components.
    useHitsSplitting = cms.bool(True),
# After in-out tracking, do out-in tracking through the seeding
# region and then further in.
    doSeedingRegionRebuilding = cms.bool(True),
#    SeedProducer = cms.string('globalMixedSeeds'),
#    SeedLabel = cms.string(''),
    maxNSeeds = cms.uint32(500000),
    maxSeedsBeforeCleaning = cms.uint32(5000),
# SeedProducer:SeedLabel descoped to src
    src = cms.InputTag('globalMixedSeeds'),                                  
    SimpleMagneticField = cms.string(''),                                    
#    SimpleMagneticField = cms.string('ParabolicMf'), # parabolic magnetic field
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryBuilder = cms.string('GroupedCkfTrajectoryBuilder'),
    TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string('GroupedCkfTrajectoryBuilder')),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite'),
#        propagatorAlongTISE = cms.string('PropagatorWithMaterialParabolicMf'),  # parabolic magnetic field
#        propagatorOppositeTISE = cms.string('PropagatorWithMaterialParabolicMfOpposite'), # parabolic magnetic field
        numberMeasurementsForFit = cms.int32(4)
    ),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
)

