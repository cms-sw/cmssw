import FWCore.ParameterSet.Config as cms

from EventFilter.CTPPSRawToDigi.totemVFATRawToDigi_cfi import totemVFATRawToDigi

ctppsDiamondRawToDigi = totemVFATRawToDigi.clone(
    subSystem = cms.string('TimingDiamond')
)
