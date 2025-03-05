import FWCore.ParameterSet.Config as cms
import os
from L1Trigger.Phase2L1ParticleFlow.l1pfJetMet_cff import L1TPFJetsExtendedTask
from L1Trigger.Phase2L1ParticleFlow.l1tSC4NGJetProducer_cfi import l1tSC4NGJetProducer


l1tSC4NGJetProducerPuppi = l1tSC4NGJetProducer.clone(
    jets = ("l1tSC4PFL1PuppiExtended", ""),
    maxJets = 16,
    minPt = 10,
    vtx = ("l1tVertexFinderEmulator","L1VerticesEmulation"),
    l1tSC4NGJetModelPath = cms.string(os.environ['CMSSW_BASE']+"/src/L1TSC4NGJetModel/L1TSC4NGJetModel"),
    # TODO correct the classes
    classes = cms.vstring(["uds", "g", "b", "c", "tau_p", "tau_n", "e", "mu"])
)


l1tSC4NGJetProducerPuppiCorrectedEmulator = l1tSC4NGJetProducerPuppi.clone(
    jets = ("l1tSC4PFL1PuppiExtendedCorrectedEmulator", "")
)

l1tSC4NGJetTask = cms.Task(
    L1TPFJetsExtendedTask, l1tSC4NGJetProducerPuppi, l1tSC4NGJetProducerPuppiCorrectedEmulator
)
