import FWCore.ParameterSet.Config as cms

from EventFilter.CTPPSRawToDigi.totemVFATRawToDigi_cfi import totemVFATRawToDigi

totemTimingRawToDigi = totemVFATRawToDigi.clone(
    subSystem = cms.string('TotemTiming'),
    
    # IMPORTANT: leave empty to load the default configuration from
    #    DataFormats/FEDRawData/interface/FEDNumbering.h
    fedIds = cms.vuint32(),
    
    RawToDigi = cms.PSet(
    verbosity = cms.untracked.uint32(0),

    # disable all the checks
    testFootprint = cms.uint32(0),
    testCRC = cms.uint32(0),
    testID = cms.uint32(0),               # compare the ID from data and mapping
    testECMostFrequent = cms.uint32(0),   # compare frame's EC with the most frequent value in the event
    testBCMostFrequent = cms.uint32(0),   # compare frame's BC with the most frequent value in the event
    
    # if non-zero, prints a per-VFAT error summary at the end of the job
    printErrorSummary = cms.untracked.uint32(0),
    
    # if non-zero, prints a summary of frames found in data, but not in the mapping
    printUnknownFrameSummary = cms.untracked.uint32(0)
  )
)
