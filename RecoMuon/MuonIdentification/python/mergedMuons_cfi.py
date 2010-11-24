import FWCore.ParameterSet.Config as cms

from RecoMuon.MuonIdentification.calomuons_cfi import calomuons
mergedMuons = cms.EDProducer("CaloMuonMerger",
    muons     = cms.InputTag("muons"), 
    muonsCut = cms.string(""),
    caloMuons = cms.InputTag("calomuons"),
    caloMuonsCut = cms.string(""),
    minCaloCompatibility = calomuons.minCaloCompatibility,
    mergeTracks = cms.bool(False),
    tracks = cms.InputTag("generalTracks"),
    tracksCut = cms.string(""),
)

