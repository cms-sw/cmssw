import FWCore.ParameterSet.Config as cms
from EventFilter.RawDataCollector.rawDataCollector_cfi import *

from EventFilter.CTPPSRawToDigi.ctppsPixelRawData_cfi import ctppsPixelRawData
from EventFilter.CTPPSRawToDigi.ctppsTotemRawData_cfi import ctppsTotemRawData
#ctppsPixelRawData.InputLabel = cms.InputTag("RPixDetDigitizer")

#ctppsPixelRawData = cms.EDProducer("CTPPSPixelDigiToRaw",
#    InputLabel = cms.InputTag("RPixDetDigitizer"),
#    mappingLabel = cms.string("RPix") 
#)

#ctppsTotemRawData = cms.EDProducer("CTPPSTotemDigiToRaw",
#    InputLabel = cms.InputTag("RPSiDetDigitizer")
#)

ctppsRawData = cms.Sequence(ctppsTotemRawData*ctppsPixelRawData)

# add CTPPS 2016 digi modules
from Configuration.Eras.Modifier_ctpps_2016_cff import ctpps_2016

_ctpps_2016_Raw = ctppsRawData.copy()
_ctpps_2016_Raw = cms.Sequence(ctppsTotemRawData*ctppsPixelRawData)
ctpps_2016.toReplaceWith(ctppsRawData,_ctpps_2016_Raw)




