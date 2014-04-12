import FWCore.ParameterSet.Config as cms
from RecoMuon.TrackingTools.MuonServiceProxy_cff import *

MuonShowerParameters = cms.PSet(
    MuonShowerInformationFillerParameters = cms.PSet(
      MuonServiceProxy,
    
      DTRecSegmentLabel = cms.InputTag("dt1DRecHits"),
      CSCRecSegmentLabel = cms.InputTag("csc2DRecHits"),
      RPCRecSegmentLabel = cms.InputTag("rpcRecHits"),
      DT4DRecSegmentLabel = cms.InputTag("dt4DSegments"),
      CSCSegmentLabel = cms.InputTag("cscSegments"),

      TrackerRecHitBuilder = cms.string('WithTrackAngle'),
      MuonRecHitBuilder = cms.string('MuonRecHitBuilder'),
    
   )
)
