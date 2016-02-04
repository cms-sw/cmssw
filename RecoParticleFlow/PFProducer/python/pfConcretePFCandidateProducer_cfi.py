import FWCore.ParameterSet.Config as cms

pfConcretePFCandidateProducer = cms.EDProducer("PFConcretePFCandidateProducer",
                                               src = cms.InputTag('particleFlow')
                                               )
