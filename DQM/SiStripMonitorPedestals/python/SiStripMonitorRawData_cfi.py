import FWCore.ParameterSet.Config as cms

RawDataMon = DQMStep1Module('SiStripMonitorRawData',
    OutputMEsInRootFile = cms.bool(False),
    DigiProducer = cms.string('siStripDigis'),
    OutputFileName = cms.string('SiStripRawData.root')
)


