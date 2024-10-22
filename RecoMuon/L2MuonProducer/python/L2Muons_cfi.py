import FWCore.ParameterSet.Config as cms
# The services
from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from RecoMuon.TrackingTools.MuonTrackLoader_cff import *
import RecoMuon.L2MuonProducer.L2MuonProducer_cfi as _mod

L2Muons = _mod.L2MuonProducer.clone(
    MuonTrackLoaderForSTA,
    MuonServiceProxy,
    InputObjects = "L2MuonSeeds",
    MuonTrajectoryBuilder = "StandAloneMuonTrajectoryBuilder", 
    SeedTransformerParameters = dict(),
    L2TrajBuilderParameters = dict(
        FilterParameters = dict(
            DTRecSegmentLabel = "dt4DSegments",
            MuonTrajectoryUpdatorParameters = dict(),
            CSCRecSegmentLabel = "cscSegments",
            RPCRecSegmentLabel = "rpcRecHits",
            Propagator = 'SteppingHelixPropagatorL2Any',
        ),
        # a precise propagation direction can be choosen accordingly with the 
        # above seed position
        SeedPropagator = 'SteppingHelixPropagatorL2Any',
        # where you want the seed (in,out)
        SeedPosition = 'in',
        BWFilterParameters = dict(
            DTRecSegmentLabel = "dt4DSegments",
            MuonTrajectoryUpdatorParameters = dict(
                Granularity = 2,
            ),
            CSCRecSegmentLabel = "cscSegments",
            RPCRecSegmentLabel = "rpcRecHits",
            Propagator = 'SteppingHelixPropagatorL2Any',
        ),
    )
)



