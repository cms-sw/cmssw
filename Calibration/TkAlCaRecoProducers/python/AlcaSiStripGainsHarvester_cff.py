import FWCore.ParameterSet.Config as cms

from Calibration.TkAlCaRecoProducers.AlcaSiStripGainsHarvester_cfi import *
from DQMServices.Components.EDMtoMEConverter_cfi import *

EDMtoMEConvertSiStripGains = EDMtoMEConverter.clone()
EDMtoMEConvertSiStripGains.lumiInputTag = cms.InputTag("MEtoEDMConvertSiStripGains","MEtoEDMConverterLumi")
EDMtoMEConvertSiStripGains.runInputTag = cms.InputTag("MEtoEDMConvertSiStripGains","MEtoEDMConverterRun")

EDMtoMEConvertSiStripGainsAllBunch = EDMtoMEConverter.clone()
EDMtoMEConvertSiStripGainsAllBunch.lumiInputTag = cms.InputTag("MEtoEDMConvertSiStripGainsAllBunch","MEtoEDMConverterLumi")
EDMtoMEConvertSiStripGainsAllBunch.runInputTag = cms.InputTag("MEtoEDMConvertSiStripGainsAllBunch","MEtoEDMConverterRun")

EDMtoMEConvertSiStripGainsAllBunch0T = EDMtoMEConverter.clone()
EDMtoMEConvertSiStripGainsAllBunch0T.lumiInputTag = cms.InputTag("MEtoEDMConvertSiStripGainsAllBunch0T","MEtoEDMConverterLumi")
EDMtoMEConvertSiStripGainsAllBunch0T.runInputTag = cms.InputTag("MEtoEDMConvertSiStripGainsAllBunch0T","MEtoEDMConverterRun")

EDMtoMEConvertSiStripGainsIsoBunch = EDMtoMEConverter.clone()
EDMtoMEConvertSiStripGainsIsoBunch.lumiInputTag = cms.InputTag("MEtoEDMConvertSiStripGainsIsoBunch","MEtoEDMConverterLumi")
EDMtoMEConvertSiStripGainsIsoBunch.runInputTag = cms.InputTag("MEtoEDMConvertSiStripGainsIsoBunch","MEtoEDMConverterRun")

EDMtoMEConvertSiStripGainsIsoBunch0T = EDMtoMEConverter.clone()
EDMtoMEConvertSiStripGainsIsoBunch0T.lumiInputTag = cms.InputTag("MEtoEDMConvertSiStripGainsIsoBunch0T","MEtoEDMConverterLumi")
EDMtoMEConvertSiStripGainsIsoBunch0T.runInputTag = cms.InputTag("MEtoEDMConvertSiStripGainsIsoBunch0T","MEtoEDMConverterRun")



DQMStore = cms.Service("DQMStore")

ALCAHARVESTSiStripGains = cms.Sequence( EDMtoMEConvertSiStripGainsAllBunch + 
                                        EDMtoMEConvertSiStripGainsAllBunch0T +
                                        EDMtoMEConvertSiStripGainsIsoBunch +
                                        EDMtoMEConvertSiStripGainsIsoBunch0T +
                                        EDMtoMEConvertSiStripGains + alcaSiStripGainsHarvester)
