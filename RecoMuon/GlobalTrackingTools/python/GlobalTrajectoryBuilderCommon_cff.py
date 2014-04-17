import FWCore.ParameterSet.Config as cms

from RecoMuon.GlobalTrackingTools.MuonTrackingRegionCommon_cff import *
from RecoMuon.GlobalTrackingTools.GlobalMuonTrackMatcher_cff import *

from RecoMuon.TransientTrackingRecHit.MuonTransientTrackingRecHitBuilder_cfi import *
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *

GlobalTrajectoryBuilderCommon = cms.PSet(
    MuonTrackingRegionCommon,
    GlobalMuonTrackMatcher,
    ScaleTECxFactor = cms.double(-1.0),
    ScaleTECyFactor = cms.double(-1.0),
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
    PCut = cms.double(2.5),
    TrackerPropagator = cms.string('SteppingHelixPropagatorAny'),
    GlbRefitterParameters = cms.PSet(
        DTRecSegmentLabel = cms.InputTag("dt4DSegments"),
        CSCRecSegmentLabel = cms.InputTag("cscSegments"),
    	GEMRecHitLabel = cms.InputTag("gemRecHits"),
        
        MuonHitsOption = cms.int32(1),
        PtCut = cms.double(1.0),
        Chi2ProbabilityCut = cms.double(30.0),
        Chi2CutCSC = cms.double(150.0),
        Chi2CutDT = cms.double(10.0),
        Chi2CutRPC = cms.double(1.0),
        Chi2CutGEM = cms.double(1.0),
        HitThreshold = cms.int32(1),
        
        Fitter = cms.string('GlbMuKFFitter'),
        Propagator = cms.string('SmartPropagatorAnyRK'),
        TrackerRecHitBuilder = cms.string('WithTrackAngle'),
        MuonRecHitBuilder = cms.string('MuonRecHitBuilder'),
        DoPredictionsOnly = cms.bool(False),
        RefitDirection = cms.string('insideOut'),
        PropDirForCosmics = cms.bool(False),
        RefitRPCHits = cms.bool(True),
        
        # only the first two are used
        DYTthrs =  cms.vint32(30, 15),

        # muon station to be skipped
        SkipStation		= cms.int32(-1),
        
        # PXB = 1, PXF = 2, TIB = 3, TID = 4, TOB = 5, TEC = 6
        TrackerSkipSystem	= cms.int32(-1),
        
        # layer, wheel, or disk depending on the system
        TrackerSkipSection	= cms.int32(-1)
        ),
)
