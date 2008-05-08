import FWCore.ParameterSet.Config as cms

from L1Trigger.CSCCommonTrigger.CSCCommonTrigger_cfi import *
csctfpacker = cms.EDFilter("CSCTFPacker",
    CSCCommonTrigger,
    # the above "using" statement is equivalent to settings below:
    #   untracked(?) int32 MinBX = 3
    #   untracked(?) int32 MaxBX = 9
    zeroSuppression = cms.bool(True),
    outputFile = cms.untracked.string(''),
    lctProducer = cms.untracked.InputTag("simCscTriggerPrimitiveDigis","MPCSORTED"),
    trackProducer = cms.untracked.InputTag("simCsctfTrackDigis"),
    putBufferToEvent = cms.untracked.bool(True),
    activeSectors = cms.int32(4095),
    nTBINs = cms.int32(7),
    # Agreement in CSC community to shift and reverse ME-1 strips as opposed to hardware
    swapME1strips = cms.bool(True)
)


