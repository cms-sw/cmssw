import FWCore.ParameterSet.Config as cms

trackAnalyzer = cms.EDAnalyzer('TrackAnalyzer',
    doTrack = cms.untracked.bool(True),
    trackPtMin = cms.untracked.double(0.01),
    trackEtaMax = cms.untracked.double(4.0),
    applyTrackSelections = cms.untracked.bool(False),
    vertexSrc = cms.InputTag("unpackedTracksAndVertices"),
    trackSrc = cms.InputTag("unpackedTracksAndVertices"),
    beamSpotSrc = cms.untracked.InputTag('offlineBeamSpot'),
    dedxEstimators = cms.VInputTag(),
)
