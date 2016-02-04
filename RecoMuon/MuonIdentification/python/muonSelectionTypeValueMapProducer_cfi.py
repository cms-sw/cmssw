import FWCore.ParameterSet.Config as cms

muonSelectionTypeValueMapProducer = cms.EDProducer("MuonSelectionTypeValueMapProducer",
    inputMuonCollection = cms.InputTag("muons1stStep"),
    selectionType = cms.string("All")
)
