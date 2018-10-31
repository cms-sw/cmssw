import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.RawToDigi_Data_cff import *
from EventFilter.RawDataCollector.rawDataMapperByLabel_cfi import rawDataMapperByLabel

RawToDigi.add(rawDataMapperByLabel)
RawToDigi_noTk.add(rawDataMapperByLabel)
RawToDigi_pixelOnly.add(rawDataMapperByLabel)

