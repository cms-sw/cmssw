import FWCore.ParameterSet.Config as cms

# L1MetPfProducer = cms.EDProducer("L1MetPfProducer",
L1MetPfProducer = cms.EDProducer("L1MetPfProducerHLS",
                                 L1PFObjects = cms.InputTag("L1PFProducer","l1pfCandidates")
)
