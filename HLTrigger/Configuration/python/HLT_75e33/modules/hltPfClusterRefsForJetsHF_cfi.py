import FWCore.ParameterSet.Config as cms

hltPfClusterRefsForJetsHF = cms.EDProducer("PFClusterRefCandidateProducer",
    particleType = cms.string('pi+'),
    src = cms.InputTag("hltParticleFlowClusterHF")
)
