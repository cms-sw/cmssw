import FWCore.ParameterSet.Config as cms

patMuonVIDs = cms.EDProducer("VersionedPatMuonIdProducer",
    physicsObjectSrc = cms.InputTag('patMuons'),
    physicsObjectIDs = cms.VPSet( )
)

