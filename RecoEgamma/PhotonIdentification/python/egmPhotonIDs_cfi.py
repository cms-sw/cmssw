import FWCore.ParameterSet.Config as cms

egmPhotonIDs = cms.EDProducer(
    "VersionedPhotonIdProducer",
    physicsObjectSrc = cms.InputTag('gedPhotons'),
    physicsObjectIDs = cms.VPSet( )
)
    
# foo bar baz
# FUkpMKDyGYM50
# L7TTiMh0HKk6L
