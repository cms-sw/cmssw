import FWCore.ParameterSet.Config as cms

genParticles = cms.EDProducer("GenParticleProducer",
    saveBarCodes = cms.untracked.bool(True),
    src = cms.InputTag("generatorSmeared"),
    abortOnUnknownPDGCode = cms.untracked.bool(False)
)


