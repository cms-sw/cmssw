import FWCore.ParameterSet.Config as cms

from L1Trigger.Phase2L1ParticleFlow.l1pfJetMet_cff import L1TPFJetsExtendedTask

from L1Trigger.Phase2L1ParticleFlow.TOoLLiPProducer_cfi import  TOoLLiPProducer
l1tTOoLLiPProducer = TOoLLiPProducer.clone(
    jets = ("l1tSC4PFL1PuppiExtended", ""),
    maxJets = 6,
    minPt = 10,
    vtx = ("l1tVertexFinderEmulator","L1VerticesEmulation")
)


l1tTOoLLiPProducerCorrectedEmulator = l1tTOoLLiPProducer.clone(
    jets = ("l1tSC4PFL1PuppiExtendedCorrectedEmulator", "")
)

L1TTOoLLiPTask = cms.Task(
    L1TPFJetsExtendedTask, l1tTOoLLiPProducer, l1tTOoLLiPProducerCorrectedEmulator
)
