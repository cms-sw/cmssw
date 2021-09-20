import FWCore.ParameterSet.Config as cms

hltPhase2L3MuonHighPtTripletStepTrackCandidates = cms.EDProducer("CkfTrackCandidateMaker",
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    SimpleMagneticField = cms.string('ParabolicMf'),
    TrajectoryBuilder = cms.string('GroupedCkfTrajectoryBuilder'),
    TrajectoryBuilderPSet = cms.PSet(
        refToPSet_ = cms.string('hltPhase2L3MuonHighPtTripletStepTrajectoryBuilder')
    ),
    TrajectoryCleaner = cms.string('hltPhase2L3MuonHighPtTripletStepTrajectoryCleanerBySharedHits'),
    TransientInitialStateEstimatorParameters = cms.PSet(
        numberMeasurementsForFit = cms.int32(4),
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    ),
    cleanTrajectoryAfterInOut = cms.bool(True),
    doSeedingRegionRebuilding = cms.bool(True),
    maxNSeeds = cms.uint32(100000),
    maxSeedsBeforeCleaning = cms.uint32(1000),
    numHitsForSeedCleaner = cms.int32(50),
    onlyPixelHitsForSeedCleaner = cms.bool(True),
    phase2clustersToSkip = cms.InputTag("hltPhase2L3MuonHighPtTripletStepClusters"),
    propagatorAlongTISE = cms.string('PropagatorWithMaterialParabolicMf'),
    propagatorOppositeTISE = cms.string('PropagatorWithMaterialParabolicMfOpposite'),
    reverseTrajectories = cms.bool(False),
    src = cms.InputTag("hltPhase2L3MuonHighPtTripletStepSeeds"),
    useHitsSplitting = cms.bool(False)
)
