import FWCore.ParameterSet.Config as cms
trackTypeMonitor = cms.EDAnalyzer('TrackTypeMonitor',
    trackInputTag   = cms.untracked.InputTag('selectedTracks'),
    offlineBeamSpot = cms.untracked.InputTag('offlineBeamSpot'),
    trackQuality    = cms.untracked.string('highPurity'),
    vertexTag       = cms.untracked.InputTag('selectedPrimaryVertices'),
    # isMC            = cms.untracked.bool(True),
    # PUCorrection    = cms.untracked.bool(False),
    TrackEtaPar    = cms.PSet(Xbins = cms.int32(60),Xmin = cms.double(-3.0),Xmax = cms.double(3.0)),
    TrackPtPar     = cms.PSet(Xbins = cms.int32(100),Xmin = cms.double(0.0),Xmax = cms.double(100.0)),
    TrackPPar      = cms.PSet(Xbins = cms.int32(100),Xmin = cms.double(0.0),Xmax = cms.double(100.0)),
    TrackPhiPar    = cms.PSet(Xbins = cms.int32(100),Xmin = cms.double(-4.0),Xmax = cms.double(4.0)),
    TrackPterrPar  = cms.PSet(Xbins = cms.int32(100),Xmin = cms.double(0.0),Xmax = cms.double(100.0)),
    TrackqOverpPar = cms.PSet(Xbins = cms.int32(100),Xmin = cms.double(-10.0),Xmax = cms.double(10.0)),
    TrackdzPar     = cms.PSet(Xbins = cms.int32(100),Xmin = cms.double(-100.0),Xmax = cms.double(100.0)),
    TrackChi2bynDOFPar = cms.PSet(Xbins = cms.int32(100),Xmin = cms.double(0.0),Xmax = cms.double(10.0)),
    nTracksPar     = cms.PSet(Xbins = cms.int32(100),Xmin = cms.double(-0.5),Xmax = cms.double(99.5))
)
