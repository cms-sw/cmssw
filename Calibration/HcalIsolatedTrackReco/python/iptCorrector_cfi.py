import FWCore.ParameterSet.Config as cms

iptCorrector = cms.EDFilter("IPTCorrector",
    corTracksLabel = cms.InputTag( "hltHITCtfWithMaterialTracks" ),
    filterLabel = cms.InputTag( "hltIsolPixelTrackL2Filter" ),
    corrIsolRadiusHB = cms.double( 0.4 ),
    corrIsolRadiusHE = cms.double( 0.5 ),
    corrIsolMaxP = cms.double( 2.0 )
)


