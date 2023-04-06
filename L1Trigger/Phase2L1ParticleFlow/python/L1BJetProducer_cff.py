import FWCore.ParameterSet.Config as cms

from L1Trigger.Phase2L1ParticleFlow.l1pfJetMet_cff import L1TPFJetsExtendedTask

from L1Trigger.Phase2L1ParticleFlow.L1BJetProducer_cfi import L1BJetProducer as _L1BJetProducer

l1tBJetProducerPuppi = _L1BJetProducer.clone(
    jets = cms.InputTag("l1tSCPFL1PuppiExtended", ""),
    maxJets = cms.int32(6),
    minPt = cms.double(10),
    vtx = cms.InputTag("l1tVertexFinderEmulator","l1verticesEmulation"),
)


l1tBJetProducerPuppiCorrectedEmulator = l1tBJetProducerPuppi.clone(
    jets = cms.InputTag("l1tSCPFL1PuppiExtendedCorrectedEmulator", ""),
)

L1TBJetsTask = cms.Task(
    L1TPFJetsExtendedTask, l1tBJetProducerPuppi, l1tBJetProducerPuppiCorrectedEmulator
)
