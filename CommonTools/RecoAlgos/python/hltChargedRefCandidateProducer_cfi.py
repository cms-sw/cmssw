import FWCore.ParameterSet.Config as cms

hltChargedRefCandidateProducer = cms.EDProducer("ChargedRefCandidateProducer",
    src          = cms.InputTag('tracks'),
    particleType = cms.string('pi+')
)
