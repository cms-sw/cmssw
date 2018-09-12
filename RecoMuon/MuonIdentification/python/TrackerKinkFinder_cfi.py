import FWCore.ParameterSet.Config as cms

TrackerKinkFinderParametersBlock = cms.PSet(
    TrackerKinkFinderParameters  = cms.PSet(
        # use also position degrees of freedom of the track state
        usePosition = cms.bool(True),
        # discard off-diagonal terms when computing chi2s
        diagonalOnly = cms.bool(False),
        # configuraton for refitter
        DoPredictionsOnly = cms.bool(False),
        Fitter = cms.string('KFFitterForRefitInsideOut'),
        TrackerRecHitBuilder = cms.string('WithAngleAndTemplate'),
        Smoother = cms.string('KFSmootherForRefitInsideOut'),
        MuonRecHitBuilder = cms.string('MuonRecHitBuilder'),
        MTDRecHitBuilder = cms.string('MTDRecHitBuilder'),
        RefitDirection = cms.string('alongMomentum'),
        RefitRPCHits = cms.bool(True),
        Propagator = cms.string('SmartPropagatorAnyRKOpposite'),
    )
)

# This customization will be removed once we get the templates for
# phase2 pixel
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(TrackerKinkFinderParametersBlock, TrackerKinkFinderParameters = dict(TrackerRecHitBuilder = 'WithTrackAngle')) # FIXME

