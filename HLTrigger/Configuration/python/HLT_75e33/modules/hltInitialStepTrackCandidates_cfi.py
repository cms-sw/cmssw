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
    lstInput = cms.InputTag('hltInputLST'),
    lstPixelSeeds = cms.InputTag('hltInputLST'),
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


_hltInitialStepTrackCandidatesMkFit = cms.EDProducer('MkFitOutputConverter',
    batchSize = cms.int32(16),
    candMVASel = cms.bool(False),
    candCutSel = cms.bool(True),
    candMinNHitsCut = cms.int32(3),
    candMinPtCut = cms.double(0.8),
    candWP = cms.double(0),
    doErrorRescale = cms.bool(True),
    mightGet = cms.optional.untracked.vstring,
    mkFitEventOfHits = cms.InputTag("hltMkFitEventOfHits"),
    mkFitPixelHits = cms.InputTag("hltMkFitSiPixelHits"),
    mkFitSeeds = cms.InputTag("hltInitialStepMkFitSeeds"),
    mkFitStripHits = cms.InputTag("hltMkFitSiPhase2Hits"),
    propagatorAlong = cms.ESInputTag("","PropagatorWithMaterial"),
    propagatorOpposite = cms.ESInputTag("","PropagatorWithMaterialOpposite"),
    qualityMaxInvPt = cms.double(100),
    qualityMaxPosErr = cms.double(100),
    qualityMaxR = cms.double(120),
    qualityMaxZ = cms.double(280),
    qualityMinTheta = cms.double(0.01),
    qualitySignPt = cms.bool(True),
    seeds = cms.InputTag("hltInitialStepSeeds"),
    tfDnnLabel = cms.string('trackSelectionTf'),
    tracks = cms.InputTag("hltInitialStepTrackCandidatesMkFit"),
    ttrhBuilder = cms.ESInputTag("","WithTrackAngle")
)

_hltInitialStepTrackCandidatesMkFitLSTSeeds = _hltInitialStepTrackCandidatesMkFit.clone(
    seeds = "hltInitialStepTrajectorySeedsLST",
    candMinNHitsCut = 4,
    candMinPtCut = 0.9
)

from Configuration.ProcessModifiers.singleIterPatatrack_cff import singleIterPatatrack
from Configuration.ProcessModifiers.trackingLST_cff import trackingLST
from Configuration.ProcessModifiers.seedingLST_cff import seedingLST
from Configuration.ProcessModifiers.hltTrackingMkFitInitialStep_cff import hltTrackingMkFitInitialStep
# All useful combinations added to make the code work as expected and for clarity
(~(singleIterPatatrack & seedingLST) & trackingLST).toReplaceWith(hltInitialStepTrackCandidates, _hltInitialStepTrackCandidatesLST)
(singleIterPatatrack & trackingLST & seedingLST).toModify(hltInitialStepTrackCandidates, src = "hltInitialStepTrajectorySeedsLST") # All LST seeds
(~seedingLST & ~trackingLST & hltTrackingMkFitInitialStep).toReplaceWith(hltInitialStepTrackCandidates, _hltInitialStepTrackCandidatesMkFit)
(singleIterPatatrack & seedingLST & trackingLST & hltTrackingMkFitInitialStep).toReplaceWith(hltInitialStepTrackCandidates, _hltInitialStepTrackCandidatesMkFitLSTSeeds)
