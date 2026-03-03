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
    seeds = cms.InputTag("hltInitialStepTrajectorySeedsLST"),
    src = cms.InputTag("hltInitialStepTrackCandidatesMkFitFit"),
    ttrhBuilder = cms.ESInputTag("","WithTrackAngle")
)

from Configuration.ProcessModifiers.trackingMkFitFit_cff import trackingMkFitFit
trackingMkFitFit.toReplaceWith(hltInitialStepTracks, _hltInitialStepTracksMkFitFit)
