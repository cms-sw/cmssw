import FWCore.ParameterSet.Config as cms

#
# This cfi should be included to run the CkfTrackCandidateMaker 
#
ckfTrackCandidates = cms.EDFilter("CkfTrackCandidateMaker",
    # set it as "none" to avoid redundant seed cleaner
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    SeedLabel = cms.string(''),
    #string RedundantSeedCleaner  = "none"
    useHitsSplitting = cms.bool(False),
    doSeedingRegionRebuilding = cms.bool(False),
    #bool   seedCleaning         = false
    SeedProducer = cms.string('globalMixedSeeds'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryBuilder = cms.string('GroupedCkfTrajectoryBuilder'),
    # nested parameter set for TransientInitialStateEstimator
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    )
)


