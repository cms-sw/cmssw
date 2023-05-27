import FWCore.ParameterSet.Config as cms

from EventFilter.CTPPSRawToDigi.totemVFATRawToDigi_cfi import totemVFATRawToDigi

totemT2Digis = totemVFATRawToDigi.clone(
    subSystem = 'TotemT2',
    RawToDigi = totemVFATRawToDigi.RawToDigi.clone(
        testID = 0, #Some ID mismatch in test sample
        testCRC = 0, # no need to test CRC for diamond frames
        testECMostFrequent = 0, # show error in the DQM and then DAQ is sending resync, no need to test in the unpacker
        testBCMostFrequent = 0, # no need to test in the unpacker, footprint test is enough
        useOlderT2TestFile = False # March '23 version with two 8-bit ID fields
    )
)
