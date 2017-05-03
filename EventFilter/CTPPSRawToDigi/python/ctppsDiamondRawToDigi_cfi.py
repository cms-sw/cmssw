import FWCore.ParameterSet.Config as cms

from EventFilter.CTPPSRawToDigi.totemVFATRawToDigi_cfi import totemVFATRawToDigi

ctppsDiamondRawToDigi = totemVFATRawToDigi.clone(
    subSystem = cms.string('TimingDiamond'),
    RawToDigi = totemVFATRawToDigi.RawToDigi.clone(
        testCRC = cms.uint32(0), # no need to test CRC for diamond frames
        testECMostFrequent = cms.uint32(0) # show error in the DQM and then DAQ is sending resync, no need to test in the unpacker
    )
)
