import FWCore.ParameterSet.Config as cms

muoMuonIDs = cms.EDProducer(
    "VersionedMuonIdProducer",
    physicsObjectSrc = cms.InputTag('muons'),
    physicsObjectIDs = cms.VPSet( )
)
    
