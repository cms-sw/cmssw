import FWCore.ParameterSet.Config as cms

from L1Trigger.Phase2L1ParticleFlow.l1pfJetMet_cff import L1TPFJetsExtendedTask

l1tBJetProducerPuppi = cms.EDProducer(
    "L1BJetProducer",
    jets = ("l1tSCPFL1PuppiExtended", ""),
    maxJets = 6,
    minPt = 10,
    vtx = ("l1tVertexFinderEmulator","l1verticesEmulation")
)


l1tBJetProducerPuppiCorrectedEmulator = l1tBJetProducerPuppi.clone(
    jets = ("l1tSCPFL1PuppiExtendedCorrectedEmulator", "")
)

L1TBJetsTask = cms.Task(
    L1TPFJetsExtendedTask, l1tBJetProducerPuppi, l1tBJetProducerPuppiCorrectedEmulator
)
