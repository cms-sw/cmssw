import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from RecoMuon.TrackingTools.MuonTrackLoader_cff import *
from RecoMuon.GlobalTrackingTools.GlobalTrajectoryBuilderCommon_cff import *
globalMuons = cms.EDProducer("GlobalMuonProducer",
    MuonServiceProxy,
    MuonTrackLoaderForGLB,
    GLBTrajBuilderParameters = cms.PSet(
        GlobalTrajectoryBuilderCommon
    ),
    TrackerCollectionLabel = cms.InputTag("generalTracks"),
    MuonCollectionLabel = cms.InputTag("standAloneMuons","UpdatedAtVtx")
)

globalMuons.GLBTrajBuilderParameters.GlobalMuonTrackMatcher.Propagator = cms.string('SmartPropagatorRK')
globalMuons.GLBTrajBuilderParameters.TrackTransformer.Propagator = cms.string('SmartPropagatorAnyRK') 


