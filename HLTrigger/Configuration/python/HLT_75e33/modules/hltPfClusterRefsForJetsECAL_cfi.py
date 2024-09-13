import FWCore.ParameterSet.Config as cms

hltPfClusterRefsForJetsECAL = cms.EDProducer("PFClusterRefCandidateProducer",
    particleType = cms.string('pi+'),
    src = cms.InputTag("hltParticleFlowClusterECAL")
)
