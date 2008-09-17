import FWCore.ParameterSet.Config as cms

process = cms.Process("PERFORMANCESUMMARYWRITE")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("Geometry.TrackerSimData.trackerSimGeometryXML_cfi")

process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.source = cms.Source("EmptyIOVSource",
    firstRun = cms.untracked.uint32(1),
    lastRun = cms.untracked.uint32(1),
    timetype = cms.string('runnumber'),
    interval = cms.uint32(1)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(2),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.string('runnumber'),
    connect = cms.string('sqlite_file:sipixelperformancesummary.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiPixelPerformanceSummaryRcd'),
        tag = cms.string('dummy')
    ))
)

process.prod = cms.EDFilter("SiPixelPerformanceSummaryBuilder")

#process.print = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.prod)
#process.ep = cms.EndPath(process.print)


