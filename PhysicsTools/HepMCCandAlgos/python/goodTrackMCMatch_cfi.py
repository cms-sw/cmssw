import FWCore.ParameterSet.Config as cms

goodTrackMCMatch = cms.EDFilter("MCTruthDeltaRMatcherNew",
    src = cms.InputTag("goodTracks"),
    distMin = cms.double(0.15),
    matchPDGId = cms.vint32(13), ## muons

    matched = cms.InputTag("genParticles")
)


