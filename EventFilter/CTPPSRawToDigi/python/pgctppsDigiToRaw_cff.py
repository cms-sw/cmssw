import FWCore.ParameterSet.Config as cms

from EventFilter.CTPPSRawToDigi.ctppsDigiToRaw_cff import *
from EventFilter.RawDataCollector.rawDataCollector_cfi import *

DigiToRaw = cms.Sequence(ctppsTotemRawData*ctppsPixelRawData*rawDataCollector)




