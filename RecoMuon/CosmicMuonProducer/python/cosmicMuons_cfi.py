import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from RecoMuon.TrackingTools.MuonTrackLoader_cff import *
from TrackingTools.GeomPropagators.StraightLinePropagator_cfi import *

MuonServiceProxy.ServiceParameters.Propagators.append('StraightLinePropagator')

cosmicMuons = cms.EDProducer("CosmicMuonProducer",
    MuonTrackLoaderForCosmic,
    MuonServiceProxy,
    TrajectoryBuilderParameters = cms.PSet(
        DTRecSegmentLabel = cms.InputTag("dt4DSegments"),
        BackwardMuonTrajectoryUpdatorParameters = cms.PSet(
            MaxChi2 = cms.double(100.0),
            RescaleError = cms.bool(False),
            RescaleErrorFactor = cms.double(1.0),
            Granularity = cms.int32(2),
            UseInvalidHits = cms.bool(False),
            ExcludeRPCFromFit = cms.bool(False)
        ),
        RPCRecSegmentLabel = cms.InputTag("rpcRecHits"),
        MuonTrajectoryUpdatorParameters = cms.PSet(
            MaxChi2 = cms.double(3000.0),
            RescaleError = cms.bool(False),
            RescaleErrorFactor = cms.double(1.0),
            Granularity = cms.int32(0),
            UseInvalidHits = cms.bool(False),
            ExcludeRPCFromFit = cms.bool(False)
        ),
        EnableRPCMeasurement = cms.bool(True),
        CSCRecSegmentLabel = cms.InputTag("cscSegments"),
        BuildTraversingMuon = cms.bool(False),
        Strict1Leg = cms.bool(False),
        EnableDTMeasurement = cms.bool(True),
        MuonSmootherParameters = cms.PSet(
            PropagatorAlong = cms.string('SteppingHelixPropagatorAny'),
            PropagatorOpposite = cms.string('SteppingHelixPropagatorAny'),
            RescalingFactor = cms.double(5.0) 
        ),
        Propagator = cms.string('SteppingHelixPropagatorAny'),
        EnableCSCMeasurement = cms.bool(True),
        MuonNavigationParameters = cms.PSet(
            Barrel = cms.bool(True),
            Endcap = cms.bool(True)
        )
    ),
    MuonSeedCollectionLabel = cms.string('CosmicMuonSeed')
)

