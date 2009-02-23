import FWCore.ParameterSet.Config as cms

GlobalMuonRefitter = cms.PSet(

    Chi2ProbabilityCut = cms.double(30.0),
    Direction = cms.int32(0),
    Chi2CutCSC = cms.double(9.0),
    Chi2CutDT = cms.double(6.0),
    Chi2CutRPC = cms.double(1.0),
    HitThreshold = cms.int32(1),
    MuonHitsOption = cms.int32(1),
    DTRecSegmentLabel = cms.InputTag("dt1DRecHits"),
    CSCRecSegmentLabel = cms.InputTag("csc2DRecHits"),
    RPCRecSegmentLabel = cms.InputTag("rpcRecHits"),

    # muon station to be skipped
    SkipStation		= cms.int32(-1),

    # PXB = 1, PXF = 2, TIB = 3, TID = 4, TOB = 5, TEC = 6
    TrackerSkipSystem	= cms.int32(-1),

    # layer, wheel, or disk depending on the system
    TrackerSkipSection	= cms.int32(-1),

    Fitter = cms.string('KFFitterForRefitInsideOut'),
    TrackerRecHitBuilder = cms.string('WithTrackAngle'),
    DoPredictionsOnly = cms.bool(False),
    Smoother = cms.string('KFSmootherForRefitInsideOut'),
    MuonRecHitBuilder = cms.string('MuonRecHitBuilder'),
    RefitDirection = cms.string('insideOut'),
    RefitRPCHits = cms.bool(True),
    PtCut = cms.double(1.0),
    Propagator = cms.string('SmartPropagatorAnyRK')
)

