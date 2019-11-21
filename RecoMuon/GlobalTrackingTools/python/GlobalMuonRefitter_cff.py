import FWCore.ParameterSet.Config as cms

GlobalMuonRefitter = cms.PSet(
    DTRecSegmentLabel = cms.InputTag("dt1DRecHits"),
    CSCRecSegmentLabel = cms.InputTag("csc2DRecHits"),
    GEMRecHitLabel = cms.InputTag("gemRecHits"),
    ME0RecHitLabel = cms.InputTag("me0Segments"),
    RPCRecSegmentLabel = cms.InputTag("rpcRecHits"),

    MuonHitsOption = cms.int32(1),
    PtCut = cms.double(1.0),
    Chi2ProbabilityCut = cms.double(30.0),
    Chi2CutCSC = cms.double(1.0),
    Chi2CutDT = cms.double(30.0),
    Chi2CutGEM = cms.double(1.0),
    Chi2CutME0 = cms.double(1.0),
    Chi2CutRPC = cms.double(1.0),
    HitThreshold = cms.int32(1),

    Fitter = cms.string('KFFitterForRefitInsideOut'),
    Smoother = cms.string('KFSmootherForRefitInsideOut'),
    Propagator = cms.string('SmartPropagatorAnyRK'),
    TrackerRecHitBuilder = cms.string('WithAngleAndTemplate'),
    MuonRecHitBuilder = cms.string('MuonRecHitBuilder'),
    DoPredictionsOnly = cms.bool(False),
    RefitDirection = cms.string('insideOut'),
    PropDirForCosmics = cms.bool(False),
    RefitRPCHits = cms.bool(True),

    # DYT stuff
    DYTthrs = cms.vint32(10, 10),
    DYTselector = cms.int32(1),
    DYTupdator = cms.bool(True),
    DYTuseAPE = cms.bool(False),
    ## Parameters for DYT threshold parametrization
    DYTuseThrsParametrization = cms.bool(True),
    DYTthrsParameters = cms.PSet(
                                  eta0p8 = cms.vdouble(1, -0.919853, 0.990742),
                                  eta1p2 = cms.vdouble(1, -0.897354, 0.987738),
                                  eta2p0 = cms.vdouble(4, -0.986855, 0.998516),
                                  eta2p2 = cms.vdouble(1, -0.940342, 0.992955),
                                  eta2p4 = cms.vdouble(1, -0.947633, 0.993762),
                                ),
    # muon station to be skipped
    SkipStation		= cms.int32(-1),

    # PXB = 1, PXF = 2, TIB = 3, TID = 4, TOB = 5, TEC = 6
    TrackerSkipSystem	= cms.int32(-1),

    # layer, wheel, or disk depending on the system
    TrackerSkipSection	= cms.int32(-1),

    RefitFlag = cms.bool( True )
)

from Configuration.Eras.Modifier_fastSim_cff import fastSim
# FastSim doesn't use Runge Kute for propagation
fastSim.toModify(GlobalMuonRefitter, Propagator = "SmartPropagatorAny")
