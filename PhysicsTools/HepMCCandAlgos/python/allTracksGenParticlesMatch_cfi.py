import FWCore.ParameterSet.Config as cms

allTracksGenParticlesMatch = cms.EDFilter("MCTruthDeltaRMatcher",
    src = cms.InputTag("allTracks"),
    distMin = cms.double(0.15),
    matchPDGId = cms.vint32(),
    matched = cms.InputTag("genParticleCandidates")
)


# foo bar baz
# GNsB5SHmPBTjx
# 8Q47XRteMjiPO
