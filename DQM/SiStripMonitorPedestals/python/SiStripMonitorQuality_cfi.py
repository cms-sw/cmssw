import FWCore.ParameterSet.Config as cms

QualityMon = cms.EDFilter("SiStripMonitorQuality",
    StripQualityLabel = cms.string('test1'),
    OutputMEsInRootFile = cms.bool(False),
    OutputFileName = cms.string('SiStripQuality.root')
)


