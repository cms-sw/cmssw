import FWCore.ParameterSet.Config as cms

displacedStandAloneMuons = cms.EDProducer("StandAloneMuonProducer",
    InputObjects = cms.InputTag("displacedMuonSeeds"),
    MuonTrajectoryBuilder = cms.string('StandAloneMuonTrajectoryBuilder'),
    STATrajBuilderParameters = cms.PSet(
        BWFilterParameters = cms.PSet(
            BWSeedType = cms.string('fromGenerator'),
            CSCRecSegmentLabel = cms.InputTag("cscSegments"),
            DTRecSegmentLabel = cms.InputTag("dt4DSegments"),
            EnableCSCMeasurement = cms.bool(True),
            EnableDTMeasurement = cms.bool(True),
            EnableGEMMeasurement = cms.bool(True),
            EnableME0Measurement = cms.bool(True),
            EnableRPCMeasurement = cms.bool(True),
            FitDirection = cms.string('outsideIn'),
            GEMRecSegmentLabel = cms.InputTag("gemRecHits"),
            ME0RecSegmentLabel = cms.InputTag("me0Segments"),
            MaxChi2 = cms.double(100.0),
            MuonTrajectoryUpdatorParameters = cms.PSet(
                ExcludeRPCFromFit = cms.bool(False),
                Granularity = cms.int32(0),
                MaxChi2 = cms.double(25.0),
                RescaleError = cms.bool(False),
                RescaleErrorFactor = cms.double(100.0),
                UseInvalidHits = cms.bool(True)
            ),
            NumberOfSigma = cms.double(3.0),
            Propagator = cms.string('SteppingHelixPropagatorAny'),
            RPCRecSegmentLabel = cms.InputTag("rpcRecHits")
        ),
        DoBackwardFilter = cms.bool(True),
        DoRefit = cms.bool(False),
        DoSeedRefit = cms.bool(False),
        FilterParameters = cms.PSet(
            CSCRecSegmentLabel = cms.InputTag("cscSegments"),
            DTRecSegmentLabel = cms.InputTag("dt4DSegments"),
            EnableCSCMeasurement = cms.bool(True),
            EnableDTMeasurement = cms.bool(True),
            EnableGEMMeasurement = cms.bool(True),
            EnableME0Measurement = cms.bool(True),
            EnableRPCMeasurement = cms.bool(True),
            FitDirection = cms.string('insideOut'),
            GEMRecSegmentLabel = cms.InputTag("gemRecHits"),
            ME0RecSegmentLabel = cms.InputTag("me0Segments"),
            MaxChi2 = cms.double(1000.0),
            MuonTrajectoryUpdatorParameters = cms.PSet(
                ExcludeRPCFromFit = cms.bool(False),
                Granularity = cms.int32(0),
                MaxChi2 = cms.double(25.0),
                RescaleError = cms.bool(False),
                RescaleErrorFactor = cms.double(100.0),
                UseInvalidHits = cms.bool(True)
            ),
            NumberOfSigma = cms.double(3.0),
            Propagator = cms.string('SteppingHelixPropagatorAny'),
            RPCRecSegmentLabel = cms.InputTag("rpcRecHits")
        ),
        NavigationType = cms.string('Standard'),
        RefitterParameters = cms.PSet(
            FitterName = cms.string('KFFitterSmootherSTA'),
            ForceAllIterations = cms.bool(False),
            MaxFractionOfLostHits = cms.double(0.05),
            NumberOfIterations = cms.uint32(3),
            RescaleError = cms.double(100.0)
        ),
        SeedPosition = cms.string('in'),
        SeedPropagator = cms.string('SteppingHelixPropagatorAny'),
        SeedTransformerParameters = cms.PSet(
            Fitter = cms.string('KFFitterSmootherSTA'),
            MuonRecHitBuilder = cms.string('MuonRecHitBuilder'),
            NMinRecHits = cms.uint32(2),
            Propagator = cms.string('SteppingHelixPropagatorAny'),
            RescaleError = cms.double(100.0),
            UseSubRecHits = cms.bool(False)
        )
    ),
    ServiceParameters = cms.PSet(
        CSCLayers = cms.untracked.bool(True),
        GEMLayers = cms.untracked.bool(True),
        ME0Layers = cms.bool(True),
        Propagators = cms.untracked.vstring(
            'SteppingHelixPropagatorAny',
            'SteppingHelixPropagatorAlong',
            'SteppingHelixPropagatorOpposite',
            'SteppingHelixPropagatorL2Any',
            'SteppingHelixPropagatorL2Along',
            'SteppingHelixPropagatorL2Opposite',
            'SteppingHelixPropagatorAnyNoError',
            'SteppingHelixPropagatorAlongNoError',
            'SteppingHelixPropagatorOppositeNoError',
            'SteppingHelixPropagatorL2AnyNoError',
            'SteppingHelixPropagatorL2AlongNoError',
            'SteppingHelixPropagatorL2OppositeNoError',
            'PropagatorWithMaterial',
            'PropagatorWithMaterialOpposite',
            'SmartPropagator',
            'SmartPropagatorOpposite',
            'SmartPropagatorAnyOpposite',
            'SmartPropagatorAny',
            'SmartPropagatorRK',
            'SmartPropagatorAnyRK',
            'StraightLinePropagator'
        ),
        RPCLayers = cms.bool(True),
        UseMuonNavigation = cms.untracked.bool(True)
    ),
    TrackLoaderParameters = cms.PSet(
        DoSmoothing = cms.bool(False),
        MuonUpdatorAtVertexParameters = cms.PSet(
            BeamSpotPositionErrors = cms.vdouble(0.1, 0.1, 5.3),
            MaxChi2 = cms.double(1000000.0),
            Propagator = cms.string('SteppingHelixPropagatorOpposite')
        ),
        Smoother = cms.string('KFSmootherForMuonTrackLoader'),
        TTRHBuilder = cms.string('WithAngleAndTemplate'),
        VertexConstraint = cms.bool(False),
        beamSpot = cms.InputTag("offlineBeamSpot")
    )
)
