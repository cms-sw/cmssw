import FWCore.ParameterSet.Config as cms

genParticles2HepMC = cms.EDProducer("GenParticles2HepMCConverter",
    genParticles = cms.InputTag("genParticles"),
    genEventInfo = cms.InputTag("generator"),
    signalParticlePdgIds = cms.vint32(),
    #signalParticlePdgIds = cms.vint32(25), ## for the Higgs analysis
)
