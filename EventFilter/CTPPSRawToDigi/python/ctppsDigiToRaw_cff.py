import FWCore.ParameterSet.Config as cms
from EventFilter.RawDataCollector.rawDataCollector_cfi import *

from EventFilter.CTPPSRawToDigi.ctppsPixelRawData_cfi import ctppsPixelRawData
from EventFilter.CTPPSRawToDigi.ctppsTotemRawData_cfi import ctppsTotemRawData

ctppsRawData = cms.Task()
# The comment lines below will be included in the next PR for Run2

#from Configuration.Eras.Modifier_ctpps_2016_cff import ctpps_2016
#ctpps_2016.toReplaceWith(ctppsRawData, cms.Task(ctppsTotemRawData))

#from Configuration.Eras.Modifier_ctpps_2017_cff import ctpps_2017
#ctpps_2017.toReplaceWith(ctppsRawData, cms.Task(ctppsTotemRawData,ctppsPixelRawData))

#from Configuration.Eras.Modifier_ctpps_2018_cff import ctpps_2018
#ctpps_2018.toReplaceWith(ctppsRawData, cms.Task(ctppsPixelRawData))

from Configuration.Eras.Modifier_ctpps_2022_cff import ctpps_2022
ctpps_2022.toReplaceWith(ctppsRawData, cms.Task(ctppsPixelRawData))
