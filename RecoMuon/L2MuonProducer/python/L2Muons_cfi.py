import FWCore.ParameterSet.Config as cms

# The services
from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from RecoMuon.TrackingTools.MuonTrackLoader_cff import *
L2Muons = cms.EDProducer("L2MuonProducer",
    MuonTrackLoaderForSTA,
    MuonServiceProxy,
    InputObjects = cms.InputTag("L2MuonSeeds"),
    L2TrajBuilderParameters = cms.PSet(
        DoRefit = cms.bool(False),
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
            Propagator = cms.string('SteppingHelixPropagatorL2Any'),
            EnableCSCMeasurement = cms.bool(True)
        ),
        # a precise propagation direction can be choosen accordingly with the 
        # above seed position
        SeedPropagator = cms.string('SteppingHelixPropagatorL2Any'),
        NavigationType = cms.string('Standard'),
        DoBackwardFilter = cms.bool(True),
        # where you want the seed (in,out)
        SeedPosition = cms.string('in'),
        BWFilterParameters = cms.PSet(
            NumberOfSigma = cms.double(3.0),
            BWSeedType = cms.string('fromGenerator'),
            FitDirection = cms.string('outsideIn'),
            DTRecSegmentLabel = cms.InputTag("dt4DSegments"),
            MaxChi2 = cms.double(25.0),
            MuonTrajectoryUpdatorParameters = cms.PSet(
                MaxChi2 = cms.double(25.0),
                RescaleError = cms.bool(False),
                RescaleErrorFactor = cms.double(100.0),
                Granularity = cms.int32(2)
            ),
            EnableRPCMeasurement = cms.bool(True),
            CSCRecSegmentLabel = cms.InputTag("cscSegments"),
            EnableDTMeasurement = cms.bool(True),
            RPCRecSegmentLabel = cms.InputTag("rpcRecHits"),
            Propagator = cms.string('SteppingHelixPropagatorL2Any'),
            EnableCSCMeasurement = cms.bool(True)
        ),
        RefitterParameters = cms.PSet(
            FitterName = cms.string('KFFitterSmootherSTA'),
            Option = cms.int32(1)
        )
    )
)



