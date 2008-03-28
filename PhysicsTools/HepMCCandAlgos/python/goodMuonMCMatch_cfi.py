import FWCore.ParameterSet.Config as cms

goodMuonMCMatch = cms.EDFilter("MCTruthDeltaRMatcherNew",
    src = cms.InputTag("goodMuons"),
    distMin = cms.double(0.15),
    matchPDGId = cms.vint32(13), ## muons

    matched = cms.InputTag("genParticles")
)


