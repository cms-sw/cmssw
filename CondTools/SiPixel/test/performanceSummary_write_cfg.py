import FWCore.ParameterSet.Config as cms

process = cms.Process("PERFORMANCESUMMARYWRITE")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("Geometry.TrackerSimData.trackerSimGeometryXML_cfi")

process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'MC_3XY_V15::All'


process.source = cms.Source("EmptyIOVSource",
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    timetype = cms.string('runnumber'),
    interval = cms.uint64(1)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(2),
        authenticationPath = cms.untracked.string('.')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:sipixelperformancesummary.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiPixelPerformanceSummaryRcd'),
        tag = cms.string('dummy')
    ))
)

process.prod = cms.EDAnalyzer("SiPixelPerformanceSummaryBuilder")

#process.print = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.prod)
#process.ep = cms.EndPath(process.print)


