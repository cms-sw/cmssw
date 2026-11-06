import FWCore.ParameterSet.Config as cms

l1tBJetProducerPuppi = cms.EDProducer("L1BJetProducer",
    NNFileName = cms.FileInPath('L1Trigger/Phase2L1ParticleFlow/data/modelTT_PUP_Off_dXY_XYCut_Graph.pb'),
    NNInput = cms.string('input:0'),
    NNOutput = cms.string('sequential/dense_2/Sigmoid'),
    jets = cms.InputTag("l1tSC4PFL1PuppiExtended"),
    maxEta = cms.double(2.4),
    maxJets = cms.int32(6),
    mightGet = cms.optional.untracked.vstring,
    minPt = cms.double(10),
    nParticles = cms.int32(10),
    useRawPt = cms.bool(True),
    vtx = cms.InputTag("l1tVertexFinderEmulator","L1VerticesEmulation")
)
