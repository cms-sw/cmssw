import FWCore.ParameterSet.Config as cms

l1tTOoLLiPProducerCorrectedEmulator = cms.EDProducer("TOoLLiPProducer",
    NNInput = cms.string('input:0'),
    NNOutput = cms.string('sequential/dense_2/Sigmoid'),
    TOoLLiPVersion = cms.string('TOoLLiP_v3'),
    jets = cms.InputTag("l1tSC4PFL1PuppiExtendedCorrectedEmulator"),
    maxEta = cms.double(2.4),
    maxJets = cms.int32(6),
    mightGet = cms.optional.untracked.vstring,
    minPt = cms.double(10),
    nParticles = cms.int32(10),
    useRawPt = cms.bool(True),
    vtx = cms.InputTag("l1tVertexFinderEmulator","L1VerticesEmulation")
)
