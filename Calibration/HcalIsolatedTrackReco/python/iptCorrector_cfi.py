import FWCore.ParameterSet.Config as cms

iptCorrector = cms.EDFilter("IPTCorrector",
    corTracksLabel = cms.InputTag("generalTracks"),
    corrIsolRadius = cms.untracked.double(0.3),
    filterLabel = cms.InputTag("hltIsolPixelTrackFilter"),
    corrIsolMaxPt = cms.untracked.double(0.9)
)


