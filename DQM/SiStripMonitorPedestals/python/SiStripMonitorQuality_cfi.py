import FWCore.ParameterSet.Config as cms

QualityMon = cms.EDFilter("SiStripMonitorQuality",
    OutputMEsInRootFile = cms.bool(False),
    dataLabel = cms.string('test1'),
    OutputFileName = cms.string('SiStripQuality.root')
)


