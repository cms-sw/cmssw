import FWCore.ParameterSet.Config as cms

l1tMETPFProducer = cms.EDProducer("L1MetPfProducer",
                                 L1PFObjects = cms.InputTag("l1tLayer1","Puppi"),
                                 maxCands = cms.int32(128),
                                 modelVersion = cms.string(""),
                                 Poly2File = cms.FileInPath("L1Trigger/Phase2L1ParticleFlow/data/met/l1met_ptphi2pxpy_poly2_v1.json"),
)

l1tMETMLProducer = cms.EDProducer("L1MetPfProducer",
                                 L1PFObjects = cms.InputTag("l1tLayer1","Puppi"),
                                 maxCands = cms.int32(100),
                                 modelVersion = cms.string("L1METML_v1"),
)
