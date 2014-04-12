import FWCore.ParameterSet.Config as cms

allTracksGenParticlesLeptonMatch = cms.EDFilter("MCTruthDeltaRMatcher",
    src = cms.InputTag("allTracks"),
    distMin = cms.double(0.15),
    matchPDGId = cms.vint32(11, 13),
    matched = cms.InputTag("genParticleCandidates")
)


