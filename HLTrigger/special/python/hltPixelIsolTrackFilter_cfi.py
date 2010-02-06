import FWCore.ParameterSet.Config as cms

hltPixelIsolTrackFilter= cms.EDFilter( "HLTPixelIsolTrackFilter",
    candTag = cms.InputTag( "hltHITIPTCorrectorHE1E31" ),
    MinPtTrack = cms.double( 20.0 ),
    MaxPtNearby = cms.double( 2.0 ),
    MaxEtaTrack = cms.double( 2.0 ),
    MinEtaTrack = cms.double( 1.4 ),
    filterTrackEnergy = cms.bool( True ),
    MinEnergyTrack = cms.double( 20.0 ),
    NMaxTrackCandidates = cms.int32( 15 ),
    DropMultiL2Event = cms.bool( False )
)

