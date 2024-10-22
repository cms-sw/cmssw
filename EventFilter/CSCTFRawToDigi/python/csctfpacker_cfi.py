import FWCore.ParameterSet.Config as cms

from L1Trigger.CSCTriggerPrimitives.CSCCommonTrigger_cfi import *
csctfpacker = cms.EDProducer("CSCTFPacker",
    CSCCommonTrigger,
    zeroSuppression = cms.bool(True),
    outputFile = cms.string(''),
    lctProducer = cms.InputTag("simCscTriggerPrimitiveDigis","MPCSORTED"),
    mbProducer  = cms.InputTag("null"),
    trackProducer = cms.InputTag("simCsctfTrackDigis"),
    putBufferToEvent = cms.bool(True),
    activeSectors = cms.int32(4095),
    nTBINs = cms.int32(7),
    # Agreement in CSC community to shift and reverse ME-1 strips as opposed to hardware
    swapME1strips = cms.bool(False)
)


