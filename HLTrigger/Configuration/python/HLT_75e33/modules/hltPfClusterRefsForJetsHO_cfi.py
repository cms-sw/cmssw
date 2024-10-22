import FWCore.ParameterSet.Config as cms

hltPfClusterRefsForJetsHO = cms.EDProducer("PFClusterRefCandidateProducer",
    particleType = cms.string('pi+'),
    src = cms.InputTag("hltParticleFlowClusterHO")
)
