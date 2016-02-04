import FWCore.ParameterSet.Config as cms

iptCorrector = cms.EDProducer("IPTCorrector",
    corTracksLabel = cms.InputTag( "hltHITCtfWithMaterialTracks" ),
    filterLabel = cms.InputTag( "hltIsolPixelTrackL2Filter" ),
    associationCone = cms.double( 0.2 )
)


