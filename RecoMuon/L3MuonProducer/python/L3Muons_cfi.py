import FWCore.ParameterSet.Config as cms

#this is a dump of the latest configuration of that module
#this is not the actual configuration of HLT
#changing this file will not change the behavior of HLT
#see the actual configuration in confDB

L3Muons = cms.EDProducer("L3MuonProducer",
    ServiceParameters = cms.PSet(
        Propagators = cms.untracked.vstring('SmartPropagatorAny',
            'SteppingHelixPropagatorAny',
            'SmartPropagator',
            'SteppingHelixPropagatorOpposite'),
        RPCLayers = cms.bool(True),
        UseMuonNavigation = cms.untracked.bool(True)
    ),
    MuonCollectionLabel = cms.InputTag("hltL2Muons","UpdatedAtVtx"),
    TrackLoaderParameters = cms.PSet(
        PutTkTrackIntoEvent = cms.untracked.bool(True),
        VertexConstraint = cms.bool(False),
        MuonSeededTracksInstance = cms.untracked.string('L2Seeded'),
        Smoother = cms.string('KFSmootherForMuonTrackLoader'),
        MuonUpdatorAtVertexParameters = cms.PSet(
            MaxChi2 = cms.double(1000000.0),
            Propagator = cms.string('SteppingHelixPropagatorOpposite'),
            BeamSpotPositionErrors = cms.vdouble(0.1, 0.1, 5.3)
        ),
        SmoothTkTrack = cms.untracked.bool(False),
        DoSmoothing = cms.bool(True),
        beamSpot = cms.InputTag("hltOfflineBeamSpot")
    ),
    L3TrajBuilderParameters = cms.PSet(
        ScaleTECxFactor = cms.double(-1.0),
        TrackerRecHitBuilder = cms.string('WithTrackAngle'),
        MuonTrackingRegionBuilder = cms.PSet(
            EtaR_UpperLimit_Par1 = cms.double(0.25),
            Eta_fixed = cms.double(0.2),
            OnDemand = cms.double(-1.0),
            Rescale_Dz = cms.double(3.0),
            Eta_min = cms.double(0.05),
            Rescale_phi = cms.double(3.0),
            EtaR_UpperLimit_Par2 = cms.double(0.15),
            DeltaZ_Region = cms.double(15.9),
            Rescale_eta = cms.double(3.0),
            PhiR_UpperLimit_Par2 = cms.double(0.2),
            vertexCollection = cms.InputTag("pixelVertices"),
            Phi_fixed = cms.double(0.2),
            DeltaR = cms.double(0.2),
            EscapePt = cms.double(1.5),
            UseFixedRegion = cms.bool(False),
            PhiR_UpperLimit_Par1 = cms.double(0.6),
            Phi_min = cms.double(0.05),
            UseVertex = cms.bool(False),
            beamSpot = cms.InputTag("hltOfflineBeamSpot")
        ),
        TrackerPropagator = cms.string('SteppingHelixPropagatorAny'),
        GlobalMuonTrackMatcher = cms.PSet(
            Pt_threshold1 = cms.double(0.0),
            DeltaDCut_3 = cms.double(15.0),
            MinP = cms.double(2.5),
            MinPt = cms.double(1.0),
            Chi2Cut_1 = cms.double(50.0),
            Pt_threshold2 = cms.double(999999999.0),
            LocChi2Cut = cms.double(0.001),
            Eta_threshold = cms.double(1.2),
            Quality_3 = cms.double(7.0),
            Quality_2 = cms.double(15.0),
            Chi2Cut_2 = cms.double(50.0),
            Chi2Cut_3 = cms.double(200.0),
            DeltaDCut_1 = cms.double(40.0),
            DeltaRCut_2 = cms.double(0.2),
            DeltaRCut_3 = cms.double(1.0),
            DeltaDCut_2 = cms.double(10.0),
            DeltaRCut_1 = cms.double(0.1),
            Quality_1 = cms.double(20.0),
            Propagator = cms.string('SmartPropagator')
        ),
        ScaleTECyFactor = cms.double(-1.0),
        tkTrajLabel = cms.InputTag("hltL3TkTracksFromL2"),
	tkTrajBeamSpot = cms.InputTag("hltOfflineBeamSpot"), # add a filter for L3 trajectory
	tkTrajMaxChi2 = cms.double(999), # add a filter for L3 trajectory
	tkTrajMaxDXYBeamSpot = cms.double(999), # add a filter for L3 trajectory
	tkTrajVertex = cms.InputTag("pixelVertices"), # add a filter for L3 trajectory
	tkTrajUseVertex = cms.bool(False), # add a filter for L3 trajectory
        MuonRecHitBuilder = cms.string('MuonRecHitBuilder'),
        RefitRPCHits = cms.bool(True),
        TrackTransformer = cms.PSet(
            DoPredictionsOnly = cms.bool(False),
            Fitter = cms.string('L3MuKFFitter'),
            TrackerRecHitBuilder = cms.string('WithTrackAngle'),
            Smoother = cms.string('KFSmootherForMuonTrackLoader'),
            MuonRecHitBuilder = cms.string('MuonRecHitBuilder'),
            RefitDirection = cms.string('insideOut'),
            RefitRPCHits = cms.bool(True),
            Propagator = cms.string('SmartPropagatorAny')
        ),
        PtCut = cms.double(1.0),
        PCut = cms.double(2.5),
        GlbRefitterParameters = cms.PSet(
            TrackerSkipSection = cms.int32(-1),
            DoPredictionsOnly = cms.bool(False),
            PropDirForCosmics = cms.bool(False),
            HitThreshold = cms.int32(1),
            MuonHitsOption = cms.int32(1),
            Chi2CutRPC = cms.double(1.0),
            Fitter = cms.string('L3MuKFFitter'),
            TrackerRecHitBuilder = cms.string('WithTrackAngle'),
            MuonRecHitBuilder = cms.string('MuonRecHitBuilder'),
            RefitDirection = cms.string('insideOut'),
            CSCRecSegmentLabel = cms.InputTag("hltCscSegments"),
            Chi2CutCSC = cms.double(150.0),
            Chi2CutDT = cms.double(10.0),
            Chi2CutGEM = cms.double(1.0),
            RefitRPCHits = cms.bool(True),
            SkipStation = cms.int32(-1),
            Propagator = cms.string('SmartPropagatorAny'),
            DTRecSegmentLabel = cms.InputTag("hltDt4DSegments"),
            GEMRecHitLabel = cms.InputTag("gemRecHits"),
            TrackerSkipSystem = cms.int32(-1)
        )
    )
)




