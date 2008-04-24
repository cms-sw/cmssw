import FWCore.ParameterSet.Config as cms

RawDataMon = cms.EDFilter("SiStripMonitorRawData",
    OutputMEsInRootFile = cms.bool(False),
    DigiProducer = cms.string('siStripDigis'),
    OutputFileName = cms.string('SiStripRawData.root')
)


