import FWCore.ParameterSet.Config as cms

hltPfClusterRefsForJetsHCAL = cms.EDProducer("PFClusterRefCandidateProducer",
    particleType = cms.string('pi+'),
    src = cms.InputTag("hltParticleFlowClusterHCAL")
)
