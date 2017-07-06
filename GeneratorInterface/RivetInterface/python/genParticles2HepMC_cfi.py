import FWCore.ParameterSet.Config as cms

genParticles2HepMC = cms.EDProducer("GenParticles2HepMCConverter",
    genParticles = cms.InputTag("genParticles"),
    #genParticles = cms.InputTag("mergedGenParticles"), # in case mergedGenParticles are created
    genEventInfo = cms.InputTag("generator"),
    signalParticlePdgIds = cms.vint32(),
    #signalParticlePdgIds = cms.vint32(25), ## for the Higgs analysis
    #signalParticlePdgIds = cms.vint32(6,-6), ## for the top quark analysis
)
