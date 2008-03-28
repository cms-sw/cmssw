import FWCore.ParameterSet.Config as cms

zDecayGenParticles = cms.EDFilter("PdgIdAndStatusCandDecaySelector",
    status = cms.vint32(3),
    src = cms.InputTag("genParticleCandidates"),
    pdgId = cms.vint32(23)
)


