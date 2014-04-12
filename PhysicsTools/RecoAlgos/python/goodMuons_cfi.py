import FWCore.ParameterSet.Config as cms

goodMuons = cms.EDFilter("MuonRefSelector",
    src = cms.InputTag("muons"),
    cut = cms.string('pt > 0')
)


