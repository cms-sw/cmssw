import FWCore.ParameterSet.Config as cms

recoMuonVIDs = cms.EDProducer("VersionedRecoMuonIdProducer",
    physicsObjectSrc = cms.InputTag('muons'),
    physicsObjectIDs = cms.VPSet( )
)

