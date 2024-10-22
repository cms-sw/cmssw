import FWCore.ParameterSet.Config as cms

process = cms.Process("CALIB")

process.MessageLogger = cms.Service("MessageLogger",
    QualityReader = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    debugModules = cms.untracked.vstring(''),
    files = cms.untracked.PSet(
        QualityReader = cms.untracked.PSet(

        )
    )
)

process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(100),
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    interval = cms.uint64(90)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

process.load("CalibTracker.SiStripESProducers.fake.SiStripQualityFakeESSource_cfi")

from CalibTracker.SiStripQuality.siStripQualityStatistics_cfi import siStripQualityStatistics
process.reader = siStripQualityStatistics.clone()

process.p = cms.Path(process.reader)

