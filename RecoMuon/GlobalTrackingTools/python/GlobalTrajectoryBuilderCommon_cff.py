import FWCore.ParameterSet.Config as cms

from RecoMuon.GlobalTrackingTools.MuonTrackingRegionCommon_cff import *
from RecoMuon.GlobalTrackingTools.GlobalMuonTrackMatcher_cff import *

from RecoMuon.TransientTrackingRecHit.MuonTransientTrackingRecHitBuilder_cfi import *
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
GlobalTrajectoryBuilderCommon = cms.PSet(
    MuonTrackingRegionCommon,
    GlobalMuonTrackMatcher,
    Direction = cms.int32(0),
    ScaleTECxFactor = cms.double(-1.0),
    ScaleTECyFactor = cms.double(-1.0),
    TrackRecHitBuilder = cms.string('WithTrackAngle'),
    TrackerRecHitBuilder = cms.string('WithTrackAngle'),
    MuonRecHitBuilder = cms.string('MuonRecHitBuilder'),
    RefitRPCHits = cms.bool(True),
    TrackTransformer = cms.PSet(
        Fitter = cms.string('KFFitterForRefitInsideOut'),
        TrackerRecHitBuilder = cms.string('WithTrackAngle'),
        Smoother = cms.string('KFSmootherForRefitInsideOut'),
        MuonRecHitBuilder = cms.string('MuonRecHitBuilder'),
        RefitDirection = cms.string('alongMomentum'),
        RefitRPCHits = cms.bool(True),
        DoPredictionsOnly = cms.bool(False)
    ),
    PtCut = cms.double(1.0),
    TrackerPropagator = cms.string('SteppingHelixPropagatorAny'),
)
