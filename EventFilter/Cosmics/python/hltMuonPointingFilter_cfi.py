import FWCore.ParameterSet.Config as cms

hltMuonPointingFilter= cms.EDFilter( "HLTMuonPointingFilter",
    SALabel = cms.string( "hltCosmicMuonBarrelOnly" ),
    PropagatorName = cms.string( "SteppingHelixPropagatorAny" ),
    radius = cms.double( 90.0 ),
    maxZ = cms.double( 280.0 )
)

