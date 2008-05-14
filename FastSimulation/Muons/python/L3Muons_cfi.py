import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from RecoMuon.TrackingTools.MuonTrackLoader_cff import *
from RecoMuon.GlobalTrackingTools.GlobalTrajectoryBuilderCommon_cff import *
L3Muons = cms.EDProducer("FastL3MuonProducer",
    MuonTrackLoaderForL3,
    MuonServiceProxy,
    L3TrajBuilderParameters = cms.PSet(
        GlobalTrajectoryBuilderCommon,
        SimulatedMuons = cms.InputTag("famosSimHits","MuonSimTracks"),
        #	InputTag tkTrajLabel = hltL3Trajectory
        #	InputTag l3SeedLabel = hltL3TrajectorySeedFromL2
        TrackerTrajectories = cms.InputTag("GlobalPixelWithMaterialTracks"),
        TkTrackBuilder = cms.string('muonCkfTrajectoryBuilder'),
        KFFitter = cms.string('L3MuKFFitter'),
        OutPropagator = cms.string('SmartPropagator')
    ),
    MuonCollectionLabel = cms.InputTag("hltL2Muons","UpdatedAtVtx")
)


