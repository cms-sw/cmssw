import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from RecoMuon.TrackingTools.MuonTrackLoader_cff import *
from RecoMuon.GlobalTrackingTools.GlobalTrajectoryBuilderCommon_cff import *
globalMuons = cms.EDProducer("GlobalMuonProducer",
    MuonServiceProxy,
    MuonTrackLoaderForGLB,
    GLBTrajBuilderParameters = cms.PSet(
        GlobalTrajectoryBuilderCommon,
        OutPropagator = cms.string('SmartPropagatorRK'),
        KFFitter = cms.string('GlbMuKFFitter')
    ),
    TrackerCollectionLabel = cms.InputTag("generalTracks"),
    MuonCollectionLabel = cms.InputTag("standAloneMuons","UpdatedAtVtx")
)


