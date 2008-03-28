import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from RecoMuon.TrackingTools.MuonTrackLoader_cff import *
cosmicMuons = cms.EDProducer("CosmicMuonProducer",
    MuonTrackLoaderForCosmic,
    MuonServiceProxy,
    TrajectoryBuilderParameters = cms.PSet(
        DTRecSegmentLabel = cms.InputTag("dt4DSegments"),
        BackwardMuonTrajectoryUpdatorParameters = cms.PSet(
            MaxChi2 = cms.double(100.0),
            RescaleError = cms.bool(False),
            RescaleErrorFactor = cms.double(1.0),
            Granularity = cms.int32(2)
        ),
        RPCRecSegmentLabel = cms.InputTag("rpcRecHits"),
        MuonTrajectoryUpdatorParameters = cms.PSet(
            MaxChi2 = cms.double(30000.0),
            RescaleError = cms.bool(False),
            RescaleErrorFactor = cms.double(1.0),
            Granularity = cms.int32(0)
        ),
        EnableRPCMeasurement = cms.untracked.bool(False),
        CSCRecSegmentLabel = cms.InputTag("cscSegments"),
        BuildTraversingMuon = cms.untracked.bool(True),
        EnableDTMeasurement = cms.untracked.bool(True),
        MuonSmootherParameters = cms.PSet(
            PropagatorAlong = cms.string('SteppingHelixPropagatorAlong'),
            PropagatorOpposite = cms.string('SteppingHelixPropagatorOpposite')
        ),
        Propagator = cms.string('SteppingHelixPropagatorAny'),
        EnableCSCMeasurement = cms.untracked.bool(True)
    ),
    MuonSeedCollectionLabel = cms.untracked.string('CosmicMuonSeed')
)


