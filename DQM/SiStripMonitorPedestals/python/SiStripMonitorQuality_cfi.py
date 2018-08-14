import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
QualityMon = DQMEDAnalyzer('SiStripMonitorQuality',
    StripQualityLabel = cms.string('test1'),
    OutputMEsInRootFile = cms.bool(False),
    OutputFileName = cms.string('SiStripQuality.root')
)


