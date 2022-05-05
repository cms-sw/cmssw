import FWCore.ParameterSet.Config as cms

pfClusterRefsForJetsHF = cms.EDProducer("PFClusterRefCandidateProducer",
    particleType = cms.string('pi+'),
    src = cms.InputTag("particleFlowClusterHF")
)
