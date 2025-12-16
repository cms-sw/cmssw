import FWCore.ParameterSet.Config as cms

hltInitialStepTracks = cms.EDProducer("TrackProducer",
    AlgorithmName = cms.string('initialStep'),
    Fitter = cms.string('FlexibleKFFittingSmoother'),
    GeometricInnerState = cms.bool(False),
    MeasurementTracker = cms.string(''),
    MeasurementTrackerEvent = cms.InputTag("hltMeasurementTrackerEvent"),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    Propagator = cms.string('RungeKuttaTrackerPropagator'),
    SimpleMagneticField = cms.string(''),
    TTRHBuilder = cms.string('WithTrackAngle'),
    TrajectoryInEvent = cms.bool(False),
    beamSpot = cms.InputTag("hltOnlineBeamSpot"),
    clusterRemovalInfo = cms.InputTag(""),
    src = cms.InputTag("hltInitialStepTrackCandidates"),
    useHitsSplitting = cms.bool(False),
    useSimpleMF = cms.bool(False)
)

hltInitialStepTracksSerialSync = hltInitialStepTracks.clone(
    src = "hltInitialStepTrackCandidatesSerialSync",
)

from Configuration.ProcessModifiers.singleIterPatatrack_cff import singleIterPatatrack
from Configuration.ProcessModifiers.trackingLST_cff import trackingLST
from Configuration.ProcessModifiers.seedingLST_cff import seedingLST

(~singleIterPatatrack & trackingLST & seedingLST).toModify(hltInitialStepTracks, src = "hltInitialStepTrackCandidates:nopLSTCsLST")

_hltInitialStepTracksMkFitFit = cms.EDProducer("MkFitOutputTrackConverter",
    measurementTrackerEvent = cms.InputTag("hltMeasurementTrackerEvent"),
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
    src = cms.InputTag("hltInitialStepTrackCandidatesMkFitFit"),
    ttrhBuilder = cms.ESInputTag("","WithTrackAngle")
)

_hltInitialStepTracksMkFitFitLSTSeeds = _hltInitialStepTracksMkFitFit.clone(seeds = "hltInitialStepTrajectorySeedsLST")

from Configuration.ProcessModifiers.hltTrackingMkFitInitialStep_cff import hltTrackingMkFitInitialStep
from Configuration.ProcessModifiers.trackingMkFitFit_cff import trackingMkFitFit

(hltTrackingMkFitInitialStep & trackingMkFitFit).toReplaceWith(hltInitialStepTracks, _hltInitialStepTracksMkFitFit)

(singleIterPatatrack & trackingLST & seedingLST & hltTrackingMkFitInitialStep & trackingMkFitFit).toReplaceWith(hltInitialStepTracks, _hltInitialStepTracksMkFitFitLSTSeeds)
