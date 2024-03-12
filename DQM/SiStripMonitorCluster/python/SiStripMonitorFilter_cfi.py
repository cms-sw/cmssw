import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
SiStripMonitorFilter = DQMEDAnalyzer('SiStripMonitorFilter',
    FilterProducer = cms.string('ClusterMTCCFilter')
)
# foo bar baz
# vwdtHbDe7fmLc
