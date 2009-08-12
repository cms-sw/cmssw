import FWCore.ParameterSet.Config as cms

iptCorrector = cms.EDFilter("IPTCorrector",
    corTracksLabel = cms.InputTag("generalTracks"),
    corrIsolRadiusHB = cms.untracked.double(0.3),
    corrIsolRadiusHE = cms.untracked.double(0.3),
    filterLabel = cms.InputTag("hltIsolPixelTrackFilter"),
    corrIsolMaxP = cms.untracked.double(2)
)


