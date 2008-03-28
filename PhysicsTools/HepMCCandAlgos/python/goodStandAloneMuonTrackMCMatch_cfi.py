import FWCore.ParameterSet.Config as cms

goodStandAloneMuonTrackMCMatch = cms.EDFilter("MCTruthDeltaRMatcherNew",
    src = cms.InputTag("goodStandAloneMuonTracks"),
    distMin = cms.double(0.15),
    matchPDGId = cms.vint32(13), ## muons

    matched = cms.InputTag("genParticles")
)


