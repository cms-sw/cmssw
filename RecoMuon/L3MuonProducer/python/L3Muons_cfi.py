import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from RecoMuon.TrackingTools.MuonTrackLoader_cff import *
from RecoMuon.GlobalTrackingTools.GlobalTrajectoryBuilderCommon_cff import *
L3Muons = cms.EDProducer("L3MuonProducer",
    MuonTrackLoaderForL3,
    MuonServiceProxy,
    L3TrajBuilderParameters = cms.PSet(
        GlobalTrajectoryBuilderCommon,
        l3SeedLabel = cms.InputTag(""), ##hltL3TrajectorySeedFromL2

        KFFitter = cms.string('L3MuKFFitter'),
        TkTrackBuilder = cms.string('muonCkfTrajectoryBuilder'),
        tkTrajLabel = cms.InputTag("hltL3Trajectory"),
        OutPropagator = cms.string('SmartPropagator')
    ),
    MuonCollectionLabel = cms.InputTag("hltL2Muons","UpdatedAtVtx")
)


