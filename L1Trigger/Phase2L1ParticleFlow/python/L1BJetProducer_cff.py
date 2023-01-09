import FWCore.ParameterSet.Config as cms

from L1Trigger.Phase2L1ParticleFlow.l1pfJetMet_cff import l1PFJetsExtendedTask

l1tBJetProducerPuppi = cms.EDProducer("L1BJetProducer",
    jets = cms.InputTag("l1tSCPFL1PuppiExtended", ""),
    useRawPt = cms.bool(True),
    NNFileName = cms.string("L1Trigger/Phase2L1ParticleFlow/data/modelTT_PUP_Off_dXY_XYCut_Graph.pb"),
    NNInput = cms.string("input:0"),
    NNOutput = cms.string("sequential/dense_2/Sigmoid"),
    maxJets = cms.int32(6),
    nParticles = cms.int32(10),
    minPt = cms.double(10),
    maxEta = cms.double(2.4),
    vtx = cms.InputTag("l1tVertexFinderEmulator","l1verticesEmulation"),
)
l1tBJetProducerPuppiCorrectedEmulator = l1tBJetProducerPuppi.clone(
    jets = cms.InputTag("l1tSCPFL1PuppiExtendedCorrectedEmulator", ""),
)

L1TBJetsTask = cms.Task(
    L1TPFJetsExtendedTask, l1tBJetProducerPuppi, l1tBJetProducerPuppiCorrectedEmulator
)
