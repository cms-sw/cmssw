import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
RawDataMon = DQMEDAnalyzer('SiStripMonitorRawData',
    OutputMEsInRootFile = cms.bool(False),
    DigiProducer = cms.string('siStripDigis'),
    OutputFileName = cms.string('SiStripRawData.root')
)


