import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
SiStripMonitorHLT = DQMEDAnalyzer('SiStripMonitorHLT',
    HLTProducer = cms.string('trigger')
)
