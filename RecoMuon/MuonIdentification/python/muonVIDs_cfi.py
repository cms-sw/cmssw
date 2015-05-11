import FWCore.ParameterSet.Config as cms

muonVIDs = cms.EDProducer("VersionedMuonIdProducer",
    physicsObjectSrc = cms.InputTag('muons'),
    physicsObjectIDs = cms.VPSet( )
)

