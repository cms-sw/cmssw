import FWCore.ParameterSet.Config as cms

genParticles2HepMC = cms.EDProducer("GenParticles2HepMCConverter",
    genParticles = cms.InputTag("genParticles"),
    #genParticles = cms.InputTag("mergedGenParticles"), # in case mergedGenParticles are created
    genEventInfo = cms.InputTag("generator"),
)
