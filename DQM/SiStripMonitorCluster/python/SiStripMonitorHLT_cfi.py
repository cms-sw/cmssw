import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
SiStripMonitorHLT = DQMEDAnalyzer('SiStripMonitorHLT',
    HLTProducer = cms.string('trigger')
)
# foo bar baz
# AHB8IpXu8UHqq
