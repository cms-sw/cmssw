import FWCore.ParameterSet.Config as cms

hltHighPtTripletStepTrackCandidates = cms.EDProducer("CkfTrackCandidateMaker",
    MeasurementTrackerEvent = cms.InputTag("hltMeasurementTrackerEvent"), 
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    TrajectoryBuilderPSet = cms.PSet(
        refToPSet_ = cms.string('highPtTripletStepTrajectoryBuilder')
    ),
    TrajectoryCleaner = cms.string('highPtTripletStepTrajectoryCleanerBySharedHits'),
    TransientInitialStateEstimatorParameters = cms.PSet(
        numberMeasurementsForFit = cms.int32(4),
        propagatorAlongTISE = cms.string('PropagatorWithMaterialParabolicMf'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialParabolicMfOpposite')
    ),
    cleanTrajectoryAfterInOut = cms.bool(True),
    doSeedingRegionRebuilding = cms.bool(True),
    maxNSeeds = cms.uint32(100000),
    maxSeedsBeforeCleaning = cms.uint32(1000),
    numHitsForSeedCleaner = cms.int32(50),
    onlyPixelHitsForSeedCleaner = cms.bool(True),
    phase2clustersToSkip = cms.InputTag("hltHighPtTripletStepClusters"),
    reverseTrajectories = cms.bool(False),
    src = cms.InputTag("hltHighPtTripletStepSeeds"),
    useHitsSplitting = cms.bool(False)
)

from Configuration.ProcessModifiers.singleIterPatatrack_cff import singleIterPatatrack
from Configuration.ProcessModifiers.trackingLST_cff import trackingLST
from Configuration.ProcessModifiers.seedingLST_cff import seedingLST

(~singleIterPatatrack & trackingLST & seedingLST).toModify(hltHighPtTripletStepTrackCandidates, src = "hltInitialStepTrackCandidates:pLSTSsLST")
