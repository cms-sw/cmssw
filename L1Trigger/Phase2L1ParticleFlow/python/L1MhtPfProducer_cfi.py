import FWCore.ParameterSet.Config as cms

L1MhtPfProducer = cms.EDProducer("L1MhtPfProducer",
                                 jets = cms.InputTag("scPFL1PuppiEmulator"),
                                 minJetPt = cms.double(30.0),
                                 maxJetEta = cms.double(2.4)
)
