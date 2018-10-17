import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.RawToDigi_Data_cff import *
from EventFilter.RawDataCollector.rawDataMapperByLabel_cfi import rawDataMapperByLabel

RawToDigi.insert(0, rawDataRemapperByLabel)
RawToDigi_noTk.insert(0, rawDataRemapperByLabel)
RawToDigi_pixelOnly.insert(0, rawDataRemapperByLabel)

