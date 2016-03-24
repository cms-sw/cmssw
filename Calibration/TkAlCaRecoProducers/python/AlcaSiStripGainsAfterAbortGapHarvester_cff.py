import FWCore.ParameterSet.Config as cms

from Calibration.TkAlCaRecoProducers.AlcaSiStripGainsAfterAbortGapHarvester_cfi import *
from DQMServices.Components.EDMtoMEConverter_cfi import *

EDMtoMEConvertSiStripGainsAfterAbortGap = EDMtoMEConverter.clone()
EDMtoMEConvertSiStripGainsAfterAbortGap.lumiInputTag = cms.InputTag("MEtoEDMConvertSiStripGainsAfterAbortGap","MEtoEDMConverterLumi")
EDMtoMEConvertSiStripGainsAfterAbortGap.runInputTag = cms.InputTag("MEtoEDMConvertSiStripGainsAfterAbortGap","MEtoEDMConverterRun")


DQMStore = cms.Service("DQMStore")

ALCAHARVESTSiStripGainsAfterAbortGap = cms.Sequence( EDMtoMEConvertSiStripGainsAfterAbortGap +
                                                     alcaSiStripGainsAfterAbortGapHarvester)
