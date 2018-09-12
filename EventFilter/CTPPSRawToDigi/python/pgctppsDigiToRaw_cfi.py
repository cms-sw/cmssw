import FWCore.ParameterSet.Config as cms

from EventFilter.CTPPSRawToDigi.ctppsDigiToRaw_cfi import *
from EventFilter.RawDataCollector.rawDataCollector_cfi import *

DigiToRaw = cms.Sequence(ctppsTotemRawData*ctppsPixelRawData*rawDataCollector)




