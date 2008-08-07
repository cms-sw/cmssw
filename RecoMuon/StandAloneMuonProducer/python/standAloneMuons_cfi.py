import FWCore.ParameterSet.Config as cms

# The services
from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from RecoMuon.TrackingTools.MuonTrackLoader_cff import *
standAloneMuons = cms.EDProducer("StandAloneMuonProducer",
    MuonTrackLoaderForSTA,
    MuonServiceProxy,
    InputObjects = cms.InputTag("MuonSeed"),
    STATrajBuilderParameters = cms.PSet(
        RefitterParameters = cms.PSet(
            FitterName = cms.string('KFFitterSmootherSTA'),
            NumberOfIterations = cms.uint32(3),
            ForceAllIterations = cms.bool(False),
            MaxFractionOfLostHits = cms.double(0.05)
        ),
        # a precise propagation direction can be choosen accordingly with the 
        # above seed position
        SeedPropagator = cms.string('SteppingHelixPropagatorAny'),
        DoRefit = cms.bool(False),
        NavigationType = cms.string('Standard'),
        SeedTransformerParameters = cms.PSet(
            MuonServiceProxy,
            Fitter = cms.string('KFFitterSmootherSTA'),
            MuonRecHitBuilder = cms.string('MuonRecHitBuilder'),
            Propagator = cms.string('SteppingHelixPropagatorAny'),
            NMinRecHits = cms.uint32(2)
        ),
        DoBackwardFilter = cms.bool(True),
        # where you want the seed (in,out)
        SeedPosition = cms.string('in'),
        BWFilterParameters = cms.PSet(
            NumberOfSigma = cms.double(3.0),
            BWSeedType = cms.string('fromGenerator'),
            FitDirection = cms.string('outsideIn'),
            DTRecSegmentLabel = cms.InputTag("dt4DSegments"),
            MaxChi2 = cms.double(100.0),
            MuonTrajectoryUpdatorParameters = cms.PSet(
                MaxChi2 = cms.double(100.0),
                RescaleError = cms.bool(False),
                RescaleErrorFactor = cms.double(100.0),
                Granularity = cms.int32(2)
            ),
            EnableRPCMeasurement = cms.bool(True),
            CSCRecSegmentLabel = cms.InputTag("cscSegments"),
            EnableDTMeasurement = cms.bool(True),
            RPCRecSegmentLabel = cms.InputTag("rpcRecHits"),
            Propagator = cms.string('SteppingHelixPropagatorAny'),
            EnableCSCMeasurement = cms.bool(True)
        ),
        DoSeedRefit = cms.bool(False),
        FilterParameters = cms.PSet(
            NumberOfSigma = cms.double(3.0),
            FitDirection = cms.string('insideOut'),
            DTRecSegmentLabel = cms.InputTag("dt4DSegments"),
            MaxChi2 = cms.double(1000.0),
            MuonTrajectoryUpdatorParameters = cms.PSet(
                MaxChi2 = cms.double(1000.0),
                RescaleError = cms.bool(False),
                RescaleErrorFactor = cms.double(100.0),
                Granularity = cms.int32(0)
            ),
            EnableRPCMeasurement = cms.bool(True),
            CSCRecSegmentLabel = cms.InputTag("cscSegments"),
            EnableDTMeasurement = cms.bool(True),
            RPCRecSegmentLabel = cms.InputTag("rpcRecHits"),
            Propagator = cms.string('SteppingHelixPropagatorAny'),
            EnableCSCMeasurement = cms.bool(True)
        )
    )
)



