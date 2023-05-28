import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Modifier_ctpps_2016_cff import ctpps_2016
from Configuration.Eras.Modifier_ctpps_2017_cff import ctpps_2017
from Configuration.Eras.Modifier_ctpps_2018_cff import ctpps_2018

from EventFilter.CTPPSRawToDigi.totemVFATRawToDigi_cfi import totemVFATRawToDigi

totemTimingRawToDigi = totemVFATRawToDigi.clone(
    subSystem = 'TotemTiming',
    
    fedIds = [586, 587],  #as declared in DataFormats/FEDRawData/interface/FEDNumbering.h
    
    RawToDigi = dict(
        verbosity = 0,

        # disable all the checks
        testFootprint = 0,
        testCRC = 0,
        testID = 0,               # compare the ID from data and mapping
        testECMostFrequent = 0,   # compare frame's EC with the most frequent value in the event
        testBCMostFrequent = 0,   # compare frame's BC with the most frequent value in the event
    
        # if non-zero, prints a per-VFAT error summary at the end of the job
        printErrorSummary = False,
    
        # if non-zero, prints a summary of frames found in data, but not in the mapping
        printUnknownFrameSummary = False
    )
)

# for Run 2 backward compatibility
(ctpps_2016 | ctpps_2017 | ctpps_2018).toModify(totemTimingRawToDigi,
fedIds = [] )
