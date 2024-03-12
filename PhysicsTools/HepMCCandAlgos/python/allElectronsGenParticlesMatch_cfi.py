import FWCore.ParameterSet.Config as cms

allElectronsGenParticlesMatch = cms.EDFilter("MCTruthDeltaRMatcher",
    src = cms.InputTag("allElectrons"),
    distMin = cms.double(0.15),
    matchPDGId = cms.vint32(11),
    matched = cms.InputTag("genParticleCandidates")
)


# foo bar baz
# B8Mm8dAdK4x9f
