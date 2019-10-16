import FWCore.ParameterSet.Config as cms
from EventFilter.RawDataCollector.rawDataCollector_cfi import *

from EventFilter.CTPPSRawToDigi.ctppsPixelRawData_cfi import ctppsPixelRawData
from EventFilter.CTPPSRawToDigi.ctppsTotemRawData_cfi import ctppsTotemRawData

ctppsRawData = cms.Sequence()

from Configuration.Eras.Modifier_ctpps_2016_cff import ctpps_2016 
ctpps_2016.toReplaceWith(ctppsRawData, cms.Sequence(ctppsTotemRawData))

from Configuration.Eras.Modifier_ctpps_2017_cff import ctpps_2017
ctpps_2017.toReplaceWith(ctppsRawData, cms.Sequence(ctppsTotemRawData*ctppsPixelRawData))

from Configuration.Eras.Modifier_ctpps_2018_cff import ctpps_2018
ctpps_2018.toReplaceWith(ctppsRawData, cms.Sequence(ctppsPixelRawData))

