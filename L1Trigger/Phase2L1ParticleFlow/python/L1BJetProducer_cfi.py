import FWCore.ParameterSet.Config as cms

L1BJetProducer = cms.EDProducer("L1BJetProducer",
                                jets = cms.InputTag("scPFL1Puppi"),
                                useRawPt = cms.bool(True),
                                NNFileName = cms.FileInPath("L1Trigger/Phase2L1ParticleFlow/data/modelTT_PUP_Off_dXY_XYCut_Graph.pb"),
                                NNInput = cms.string("input:0"),
                                NNOutput = cms.string("sequential/dense_2/Sigmoid"),
                                maxJets = cms.int32(10),
                                nParticle = cms.int32(10),
                                minPt = cms.double(20),
                                maxEta = cms.double(2.4),
                                vtx = cms.InputTag("L1VertexFinderEmulator", "l1verticesEmulation"),
)