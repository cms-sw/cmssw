import FWCore.ParameterSet.Config as cms

l1tMHTPFProducer = cms.EDProducer("L1MhtPfProducer",
                                 jets = cms.InputTag("l1tSCPFL1PuppiEmulator"),
                                 minJetPt = cms.double(30.0),
                                 maxJetEta = cms.double(550 * cms.math.pi / 720) # 2.4 rounded to hardware unit granularity
)
