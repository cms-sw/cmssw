import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from RecoMuon.TrackingTools.MuonTrackLoader_cff import *
from RecoMuon.GlobalTrackingTools.GlobalTrajectoryBuilderCommon_cff import *
from RecoMuon.GlobalTrackingTools.GlobalMuonRefitter_cff import *
tevMuons = cms.EDProducer("TevMuonProducer",
    MuonTrackLoaderForGLB,
    #    InputTag MuonCollectionLabel = standAloneMuons:UpdatedAtVtx
    MuonServiceProxy,
    MuonCollectionLabel = cms.InputTag("globalMuons"),
    Cocktails = cms.vstring('default', 
        'firstHit', 
        'picky'),
    CocktailIndex = cms.vint32(1, 2, 3),
    RefitterParameters = cms.PSet(
        GlobalMuonRefitter
    )
)


