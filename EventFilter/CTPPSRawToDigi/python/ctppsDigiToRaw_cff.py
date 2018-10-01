import FWCore.ParameterSet.Config as cms
from EventFilter.RawDataCollector.rawDataCollector_cfi import *

from EventFilter.CTPPSRawToDigi.ctppsPixelRawData_cfi import ctppsPixelRawData
from EventFilter.CTPPSRawToDigi.ctppsTotemRawData_cfi import ctppsTotemRawData

ppsRawData = cms.Sequence()

from Configuration.Eras.Modifier_pps_2016_cff import pps_2016
_pps_2016_Raw = ppsRawData.copy()
_pps_2016_Raw = cms.Sequence(ctppsTotemRawData)
pps_2016.toReplaceWith(ppsRawData,_pps_2016_Raw)

from Configuration.Eras.Modifier_pps_2017_cff import pps_2017
_pps_2017_Raw = ppsRawData.copy()
_pps_2017_Raw = cms.Sequence(ctppsTotemRawData*ctppsPixelRawData)
pps_2017.toReplaceWith(ppsRawData,_pps_2017_Raw)

from Configuration.Eras.Modifier_pps_2018_cff import pps_2018
_pps_2018_Raw = ppsRawData.copy()
_pps_2018_Raw = cms.Sequence(ctppsPixelRawData)
pps_2018.toReplaceWith(ppsRawData,_pps_2018_Raw)

