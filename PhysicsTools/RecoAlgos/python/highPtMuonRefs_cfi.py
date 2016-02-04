import FWCore.ParameterSet.Config as cms

highPtMuonRefs = cms.EDFilter("MuonRefSelector",
    src = cms.InputTag("muons"),
    cut = cms.string('pt > 20')
)


