import FWCore.ParameterSet.Config as cms

l1tMETPFProducer = cms.EDProducer("L1MetPfProducer",
                                 L1PFObjects = cms.InputTag("l1tLayer1","Puppi"),
                                 maxCands = cms.int32(128),
)
# foo bar baz
# 8MeqQeloyV56H
# z6dk9T5pwTTQV
