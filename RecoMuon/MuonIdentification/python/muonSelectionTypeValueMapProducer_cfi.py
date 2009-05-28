import FWCore.ParameterSet.Config as cms

muonSelectionTypeValueMapProducer = cms.EDProducer("MuonSelectionTypeValueMapProducer",
    inputMuonCollection = cms.InputTag("muons"),
    selectionType = cms.string("All")
)
