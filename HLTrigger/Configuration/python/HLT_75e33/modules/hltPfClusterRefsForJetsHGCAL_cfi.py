import FWCore.ParameterSet.Config as cms

hltPfClusterRefsForJetsHGCAL = cms.EDProducer("PFClusterRefCandidateProducer",
    particleType = cms.string('pi+'),
    src = cms.InputTag("hltParticleFlowClusterHGCal")
)
