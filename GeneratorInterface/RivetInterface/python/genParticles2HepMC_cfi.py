import FWCore.ParameterSet.Config as cms

generator = cms.EDProducer("GenParticles2HepMCConverter",
    VertexSmearing = cms.PSet(refToPSet_ = cms.string("VertexSmearingParameters")),
    genParticles = cms.InputTag("genParticles"),
    genEventInfo = cms.InputTag("generator"),
)
