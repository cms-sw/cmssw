import FWCore.ParameterSet.Config as cms

from EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi import *
SiPixelRawToDigi = cms.Sequence(siPixelDigis)
siPixelDigis.InputLabel = 'rawDataCollector'

