import FWCore.ParameterSet.Config as cms

allMuonsGenParticlesMatch = cms.EDFilter("MCTruthDeltaRMatcher",
    src = cms.InputTag("allMuons"),
    distMin = cms.double(0.15),
    matchPDGId = cms.vint32(13),
    matched = cms.InputTag("genParticleCandidates")
)


