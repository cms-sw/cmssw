import FWCore.ParameterSet.Config as cms

from Calibration.TkAlCaRecoProducers.AlcaSiStripGainsAAGHarvester_cfi import *
from DQMServices.Components.EDMtoMEConverter_cfi import *

EDMtoMEConvertSiStripGainsAAG = EDMtoMEConverter.clone()
EDMtoMEConvertSiStripGainsAAG.lumiInputTag = cms.InputTag("MEtoEDMConvertSiStripGainsAAG","MEtoEDMConverterLumi")
EDMtoMEConvertSiStripGainsAAG.runInputTag = cms.InputTag("MEtoEDMConvertSiStripGainsAAG","MEtoEDMConverterRun")

DQMStore = cms.Service("DQMStore")

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
dqmEnvSiStripGainsAAG = DQMEDAnalyzer('DQMEventInfo',
                                       subSystemFolder = cms.untracked.string('AlCaReco'),  
                                       )

ALCAHARVESTSiStripGainsAAG = cms.Sequence( EDMtoMEConvertSiStripGainsAAG + alcaSiStripGainsAAGHarvester + dqmEnvSiStripGainsAAG )
