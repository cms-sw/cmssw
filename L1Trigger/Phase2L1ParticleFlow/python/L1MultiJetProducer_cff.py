import FWCore.ParameterSet.Config as cms
import os
from L1Trigger.Phase2L1ParticleFlow.l1pfJetMet_cff import L1TPFJetsExtendedTask

from L1Trigger.Phase2L1ParticleFlow.L1MultiJetProducer_cfi import  L1MultiJetProducer
l1tMultiJetProducerPuppi = L1MultiJetProducer.clone(
    jets = ("l1tSC4PFL1PuppiExtended", ""),
    maxJets = 16,
    minPt = 10,
    vtx = ("l1tVertexFinderEmulator","L1VerticesEmulation"),
    MultiJetPath = cms.string(os.environ['CMSSW_BASE']+"/src/hls4ml-jettagger/MultiJetBaseline")
)


l1tMultiJetProducerPuppiCorrectedEmulator = l1tMultiJetProducerPuppi.clone(
    jets = ("l1tSC4PFL1PuppiExtendedCorrectedEmulator", "")
)

L1TMultiJetsTask = cms.Task(
    L1TPFJetsExtendedTask, l1tMultiJetProducerPuppi, l1tMultiJetProducerPuppiCorrectedEmulator
)
