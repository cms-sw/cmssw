import FWCore.ParameterSet.Config as cms

l1tMHTPFProducer = cms.EDProducer("L1MhtPfProducer",
                                 jets = cms.InputTag("l1tSC4PFL1PuppiEmulator"),
                                 minJetPt = cms.double(30.0),
                                 maxJetEta = cms.double(2.4)
)
