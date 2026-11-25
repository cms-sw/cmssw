import FWCore.ParameterSet.Config as cms

hltInitialStepTracks = cms.EDProducer("TrackProducer",
    AlgorithmName = cms.string('initialStep'),
    Fitter = cms.string('hltESPFlexibleKFFittingSmootherForLSTStep'),
    GeometricInnerState = cms.bool(False),
    MeasurementTracker = cms.string('hltESPMeasurementTracker'),
    MeasurementTrackerEvent = cms.InputTag("hltMeasurementTrackerEvent"),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    Propagator = cms.string('hltESPRungeKuttaTrackerPropagator'),
    SimpleMagneticField = cms.string(''),
    TTRHBuilder = cms.string('hltESPTTRHBuilderWithTrackAngle'),
    TrajectoryInEvent = cms.bool(False),
    beamSpot = cms.InputTag("hltOnlineBeamSpot"),
    clusterRemovalInfo = cms.InputTag(""),
    src = cms.InputTag("hltInitialStepTrackCandidates"),
    useHitsSplitting = cms.bool(False),
    useSimpleMF = cms.bool(False)
)

from Configuration.ProcessModifiers.mtd_at_hlt_cff import mtd_at_hlt
mtd_at_hlt.toModify(hltInitialStepTracks, TrajectoryInEvent = True)

# The outlier-tolerant fit is for the LST + mkFit step; the other initial-step configurations keep the default fit.
from Configuration.ProcessModifiers.trackingLST_cff import trackingLST
from Configuration.ProcessModifiers.hltPhase2LegacyTracking_cff import hltPhase2LegacyTracking
(trackingLST | hltPhase2LegacyTracking).toModify(hltInitialStepTracks, Fitter = 'hltESPFlexibleKFFittingSmoother')

_hltInitialStepTracksMkFitFit = cms.EDProducer("MkFitOutputTrackConverter",
    mkFitEventOfHits = cms.InputTag("hltMkFitEventOfHits"),
    mkFitPixelHits = cms.InputTag("hltMkFitSiPixelHits"),
    mkFitStripHits = cms.InputTag("hltMkFitSiPhase2Hits"),
    mkFitSeeds = cms.InputTag("hltInitialStepMkFitSeeds"),
    src = cms.InputTag("hltInitialStepTrackCandidatesMkFitFit"),
    seeds = cms.InputTag("hltInitialStepTrajectorySeedsLST"),
    algorithmName = cms.string('initialStep'),
    beamSpot = cms.InputTag("hltOnlineBeamSpot"),
    ttrhBuilder = cms.ESInputTag('', 'hltESPTTRHBuilderWithTrackAngle'),
    propagatorAlong = cms.ESInputTag('', 'PropagatorWithMaterial'),
    propagatorOpposite = cms.ESInputTag('', 'PropagatorWithMaterialOpposite'),
    qualityMaxInvPt = cms.double(100),
    qualityMinTheta = cms.double(0.01),
    qualityMaxR = cms.double(120),
    qualityMaxZ = cms.double(280),
    qualityMaxPosErr = cms.double(100),
    qualitySignPt = cms.bool(True),
    NavigationSchool = cms.ESInputTag('', 'SimpleNavigationSchool'),
    trajectoryInEvent = cms.bool(False),
    measurementTrackerEvent = cms.InputTag("hltMeasurementTrackerEvent"),
    mightGet = cms.optional.untracked.vstring
)

from Configuration.ProcessModifiers.trackingMkFitFit_cff import trackingMkFitFit
trackingMkFitFit.toReplaceWith(hltInitialStepTracks, _hltInitialStepTracksMkFitFit)
(trackingMkFitFit & mtd_at_hlt).toModify(hltInitialStepTracks, trajectoryInEvent = True)
