import FWCore.ParameterSet.Config as cms

iptCorrector = cms.EDProducer("IPTCorrector",
    corTracksLabel = cms.InputTag( "hltIter0PFlowCtfWithMaterialTracks" ),
    filterLabel = cms.InputTag( "hltIsolPixelTrackL2Filter" ),
    associationCone = cms.double( 0.2 )
)


