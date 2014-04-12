import FWCore.ParameterSet.Config as cms

QualityMon = cms.EDAnalyzer("SiStripMonitorQuality",
    StripQualityLabel = cms.string('test1'),
    OutputMEsInRootFile = cms.bool(False),
    OutputFileName = cms.string('SiStripQuality.root')
)


