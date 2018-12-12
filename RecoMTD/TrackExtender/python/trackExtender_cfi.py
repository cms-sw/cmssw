import FWCore.ParameterSet.Config as cms

from RecoMTD.TrackExtender.PropagatorWithMaterialForMTD_cfi import *

_mtdRecHitBuilder = cms.string('MTDRecHitBuilder')
_mtdPropagator = cms.string('PropagatorWithMaterialForMTD')

mtdTrackExtender = cms.EDProducer(
    'TrackExtenderWithMTD',
    tracksSrc = cms.InputTag("generalTracks"),
    hitsSrc = cms.InputTag("mtdTrackingRecHits"),
    beamSpotSrc = cms.InputTag("offlineBeamSpot"),
    updateTrackTrajectory = cms.bool(True),
    updateTrackExtra = cms.bool(True),
    updateTrackHitPattern = cms.bool(True),
    TransientTrackBuilder = cms.string('TransientTrackBuilder'),
    MTDRecHitBuilder = _mtdRecHitBuilder,
    Propagator = _mtdPropagator,
    TrackTransformer = cms.PSet(
        DoPredictionsOnly = cms.bool(False),
        Fitter = cms.string('KFFitterForRefitInsideOut'),
        #TrackerRecHitBuilder = cms.string('WithTrackAngleAndTemplate'),
        TrackerRecHitBuilder = cms.string('WithTrackAngle'),
        Smoother = cms.string('KFSmootherForRefitInsideOut'),
        MuonRecHitBuilder = cms.string('MuonRecHitBuilder'),
        MTDRecHitBuilder = _mtdRecHitBuilder,
        RefitDirection = cms.string('alongMomentum'),
        RefitRPCHits = cms.bool(True),
        Propagator = _mtdPropagator     
        )
    )
