import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Modifier_ctpps_2016_cff import ctpps_2016
from Configuration.Eras.Modifier_ctpps_2017_cff import ctpps_2017
from Configuration.Eras.Modifier_ctpps_2018_cff import ctpps_2018
from Configuration.Eras.Modifier_ctpps_2023_cff import ctpps_2022

from EventFilter.CTPPSRawToDigi.totemVFATRawToDigi_cfi import totemVFATRawToDigi

ctppsDiamondRawToDigi = totemVFATRawToDigi.clone(
    subSystem = 'TimingDiamond',
    fedIds = [579, 581, 582, 583, 588, 589], #as declared in DataFormats/FEDRawData/interface/FEDNumbering.h
    RawToDigi = dict(
        testCRC = 0,                     # no need to test CRC for diamond frames
        testECMostFrequent = 0, # show error in the DQM and then DAQ is sending resync, no need to test in the unpacker
    )
)

# for Run 2 backward compatibility
(ctpps_2016 | ctpps_2017 | ctpps_2018).toModify(ctppsDiamondRawToDigi, fedIds = [] )
# Run 3 , year 2022
ctpps_2022.toModify(ctppsDiamondRawToDigi, fedIds = [579, 581, 582, 583] )
