import FWCore.ParameterSet.Config as cms

#
# This cfi should be included to run the CkfTrajectoryMaker 
#
ckfTrajectories = cms.EDFilter("CkfTrajectoryMaker",
    # set it as "none" to avoid redundant seed cleaner
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    #string RedundantSeedCleaner  = "none"
    useHitsSplitting = cms.bool(False),
    doSeedingRegionRebuilding = cms.bool(False),
    trackCandidateAlso = cms.bool(False),
    #bool   seedCleaning         = false
    src = cms.InputTag('globalMixedSeeds'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryBuilder = cms.string('GroupedCkfTrajectoryBuilder'),
    # nested parameter set for TransientInitialStateEstimator
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite'),
        numberMeasurementsForFit = cms.int32(4)
    )
)


