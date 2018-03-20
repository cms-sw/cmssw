import FWCore.ParameterSet.Config as cms

from EventFilter.CTPPSRawToDigi.totemVFATRawToDigi_cfi import totemVFATRawToDigi

totemTimingRawToDigi = totemVFATRawToDigi.clone(
    subSystem = cms.string('TotemTiming'),
    
    # IMPORTANT: leave empty to load the default configuration from
    #    DataFormats/FEDRawData/interface/FEDNumbering.h
    fedIds = cms.vuint32(),
)
