import FWCore.ParameterSet.Config as cms

pfClusterRefsForJetsECAL = cms.EDProducer("PFClusterRefCandidateProducer",
    particleType = cms.string('pi+'),
    src = cms.InputTag("particleFlowClusterECAL")
)
