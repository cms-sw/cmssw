import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.RawToDigi_Data_cff import *
from EventFilter.RawDataCollector.rawDataMapperByLabel_cfi import rawDataMapperByLabel

RawToDigiTask.add(rawDataMapperByLabel)
RawToDigiTask_noTk.add(rawDataMapperByLabel)
RawToDigiTask_pixelOnly.add(rawDataMapperByLabel)

