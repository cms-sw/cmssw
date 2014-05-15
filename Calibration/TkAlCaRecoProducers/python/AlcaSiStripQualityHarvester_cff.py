import FWCore.ParameterSet.Config as cms

from Calibration.TkAlCaRecoProducers.AlcaSiStripQualityHarvester_cfi import *
from DQMServices.Components.EDMtoMEConverter_cfi import *

EDMtoMEConvertSiStrip = EDMtoMEConverter.clone()
EDMtoMEConvertSiStrip.lumiInputTag = cms.InputTag("MEtoEDMConvertSiStrip","MEtoEDMConverterLumi")
EDMtoMEConvertSiStrip.runInputTag = cms.InputTag("MEtoEDMConvertSiStrip","MEtoEDMConverterRun")

DQMStore = cms.Service("DQMStore")

ALCAHARVESTSiStripQuality = cms.Sequence(EDMtoMEConvertSiStrip + alcaSiStripQualityHarvester)
#ALCAHARVESTSiStripQuality = cms.Sequence(EDMtoMEConvertSiStrip + dqmSaver)
