import FWCore.ParameterSet.Config as cms

from EventFilter.CTPPSRawToDigi.totemVFATRawToDigi_cfi import totemVFATRawToDigi

totemT2Digis = totemVFATRawToDigi.clone(
    subSystem = cms.string('TotemT2'),
    RawToDigi = totemVFATRawToDigi.RawToDigi.clone(
        testID = cms.uint32(0), #Some ID mismatch in test sample
        testCRC = cms.uint32(0), # no need to test CRC for diamond frames
        testECMostFrequent = cms.uint32(0) # show error in the DQM and then DAQ is sending resync, no need to test in the unpacker
    )
)
