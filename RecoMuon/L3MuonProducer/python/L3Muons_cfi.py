import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from RecoMuon.TrackingTools.MuonTrackLoader_cff import *
from RecoMuon.GlobalTrackingTools.GlobalTrajectoryBuilderCommon_cff import *
L3Muons = cms.EDProducer("L3MuonProducer",
    MuonTrackLoaderForL3,
    MuonServiceProxy,
    L3TrajBuilderParameters = cms.PSet(
        GlobalTrajectoryBuilderCommon,
        SeedGeneratorParameters = cms.PSet(
            ComponentName = cms.string('TSGFromOrderedHits'),
            OrderedHitsFactoryPSet = cms.PSet(
                ComponentName = cms.string('StandardHitPairGenerator'),
                SeedingLayers = cms.string('PixelLayerPairs')
            ),
            TTRHBuilder = cms.string('WithTrackAngle')
        ),
        l3SeedLabel = cms.InputTag("hltL3TrajectorySeedFromL2"),
        StateOnTrackerBoundOutPropagator = cms.string('SmartPropagatorAny'),
        KFFitter = cms.string('L3MuKFFitter'),
        TkTrackBuilder = cms.string('muonCkfTrajectoryBuilder'),
        tkTrajLabel = cms.InputTag("hltL3Trajectory")
    ),
    MuonCollectionLabel = cms.InputTag("hltL2Muons","UpdatedAtVtx")
)


