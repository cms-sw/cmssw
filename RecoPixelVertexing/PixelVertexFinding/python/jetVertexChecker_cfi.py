import FWCore.ParameterSet.Config as cms
jetVertexChecker = cms.EDFilter('JetVertexChecker',
    jetTracks = cms.InputTag( "hltBLifetimeL25AssociatorHbbFirst" ),
    minPt = cms.double( 0 ),
    minPtRatio = cms.double( 0.05 ),
    doFilter = cms.bool( False ),
    beamSpot = cms.InputTag("hltOnlineBeamSpot"),
    maxNJetsToCheck = cms.int32(2)
)



