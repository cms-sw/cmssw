import FWCore.ParameterSet.Config as cms
from EventFilter.RawDataCollector.rawDataCollector_cfi import *

from EventFilter.CTPPSRawToDigi.ctppsPixelRawData_cfi import ctppsPixelRawData
from EventFilter.CTPPSRawToDigi.ctppsTotemRawData_cfi import ctppsTotemRawData

ppsRawData = cms.Sequence()

from Configuration.Eras.Modifier_pps_2016_cff import pps_2016 
pps_2016.toReplaceWith(ppsRawData, cms.Sequence(ctppsTotemRawData))

from Configuration.Eras.Modifier_pps_2017_cff import pps_2017
pps_2017.toReplaceWith(ppsRawData, cms.Sequence(ctppsTotemRawData*ctppsPixelRawData))

from Configuration.Eras.Modifier_pps_2018_cff import pps_2018
pps_2018.toReplaceWith(ppsRawData, cms.Sequence(ctppsPixelRawData))

