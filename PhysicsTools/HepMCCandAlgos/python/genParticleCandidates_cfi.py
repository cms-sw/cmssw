import FWCore.ParameterSet.Config as cms

genParticleCandidates = cms.EDProducer("FastGenParticleCandidateProducer",
    saveBarCodes = cms.untracked.bool(False),
    src = cms.InputTag("generatorSmeared"),
    abortOnUnknownPDGCode = cms.untracked.bool(False)
)


