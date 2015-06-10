import FWCore.ParameterSet.Config as cms

egmPhotonIDs = cms.EDProducer(
    "VersionedPhotonIdProducer",
    physicsObjectSrc = cms.InputTag('gedPhotons'),
    physicsObjectIDs = cms.VPSet( )
)
    
