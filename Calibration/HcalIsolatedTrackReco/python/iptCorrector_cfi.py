import FWCore.ParameterSet.Config as cms

iptCorrector = cms.EDFilter("IPTCorrector",
    corTracksLabel = cms.InputTag( "hltHITCtfWithMaterialTracks" ),
    filterLabel = cms.InputTag( "hltIsolPixelTrackL2Filter" ),
    associationCone = cms.double( 0.2 )
)


