import FWCore.ParameterSet.Config as cms

from L1Trigger.CSCCommonTrigger.CSCCommonTrigger_cfi import *
csctfunpacker = cms.EDProducer("CSCTFUnpacker",
    CSCCommonTrigger,
    # Set all values to 0 if you trust hardware settings
    # Keep in mind that +Z (positive endcap) has sectors 1-6 and -Z (negative endcap) 7-12
    slot2sector = cms.vint32(0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0),
    # Mapping file (default one-to-one applied if empty):
    mappingFile = cms.string(''),
    # Agreement in CSC community to shift and reverse ME-1 strips as opposed to hardware
    swapME1strips = cms.bool(False),
    # the above "using" statement is equivalent to setting of LCT time window below:
    #   int32 MinBX = 3
    #   int32 MaxBX = 9
    # Specify label of the module which produces raw CSCTF data
    producer = cms.InputTag("rawDataCollector")
)


