import FWCore.ParameterSet.Config as cms

pfClusterRefsForJetsHGCAL = cms.EDProducer("PFClusterRefCandidateProducer",
    particleType = cms.string('pi+'),
    src = cms.InputTag("particleFlowClusterHGCal")
)
