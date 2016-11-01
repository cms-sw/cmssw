import FWCore.ParameterSet.Config as cms

from EventFilter.CTPPSRawToDigi.totemVFATRawToDigi_cfi import totemVFATRawToDigi

ctppsDiamondRawToDigi = totemVFATRawToDigi.copy()
ctppsDiamondRawToDigi.subSystem = "TimingDiamond"
