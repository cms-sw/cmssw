import FWCore.ParameterSet.Config as cms

pfClusterRefsForJetsHCAL = cms.EDProducer("PFClusterRefCandidateProducer",
    particleType = cms.string('pi+'),
    src = cms.InputTag("particleFlowClusterHCAL")
)
