import FWCore.ParameterSet.Config as cms

hltInitialStepTrackCandidates = cms.EDProducer("CkfTrackCandidateMaker",
    MeasurementTrackerEvent = cms.InputTag("hltMeasurementTrackerEvent"),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    TrajectoryBuilderPSet = cms.PSet(
        refToPSet_ = cms.string('initialStepTrajectoryBuilder')
    ),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
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
    reverseTrajectories = cms.bool(False),
    src = cms.InputTag("hltInitialStepSeeds"),
    useHitsSplitting = cms.bool(False)
)

_hltInitialStepTrackCandidatesLST = cms.EDProducer('LSTOutputConverter',
    lstOutput = cms.InputTag('hltLST'),
    phase2OTHits = cms.InputTag('hltPhase2OTHitsInputLST'),
    lstPixelSeeds = cms.InputTag('hltPixelSeedInputLST'),
    includeT5s = cms.bool(True),
    includeNonpLSTSs = cms.bool(False),
    propagatorAlong = cms.ESInputTag('', 'PropagatorWithMaterial'),
    propagatorOpposite = cms.ESInputTag('', 'PropagatorWithMaterialOpposite'),
    SeedCreatorPSet = cms.PSet(
        ComponentName = cms.string('SeedFromConsecutiveHitsCreator'),
        propagator = cms.string('PropagatorWithMaterial'),
        SeedMomentumForBOFF = cms.double(5),
        OriginTransverseErrorMultiplier = cms.double(1),
        MinOneOverPtError = cms.double(1),
        magneticField = cms.string(''),
        TTRHBuilder = cms.string('WithTrackAngle'),
        forceKinematicWithRegionDirection = cms.bool(False)
    )
)

from Configuration.ProcessModifiers.singleIterPatatrack_cff import singleIterPatatrack
from Configuration.ProcessModifiers.trackingLST_cff import trackingLST
from Configuration.ProcessModifiers.seedingLST_cff import seedingLST
# All useful combinations added to make the code work as expected and for clarity
(~singleIterPatatrack & trackingLST & ~seedingLST).toReplaceWith(hltInitialStepTrackCandidates, _hltInitialStepTrackCandidatesLST)
(~singleIterPatatrack & trackingLST & seedingLST).toReplaceWith(hltInitialStepTrackCandidates, _hltInitialStepTrackCandidatesLST)
(singleIterPatatrack & trackingLST & ~seedingLST).toReplaceWith(hltInitialStepTrackCandidates, _hltInitialStepTrackCandidatesLST)
(singleIterPatatrack & trackingLST & seedingLST).toModify(hltInitialStepTrackCandidates, src = "hltInitialStepTrajectorySeedsLST") # All LST seeds
