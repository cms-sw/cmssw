import FWCore.ParameterSet.Config as cms

process = cms.Process("PERFORMANCESUMMARYREAD")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(10),
    firstRun = cms.untracked.uint32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(2),
        authenticationPath = cms.untracked.string('.')
    ),
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('SiPixelPerformanceSummaryRcd'),
        tag = cms.string('dummy')
    )),
    connect = cms.string('sqlite_file:sipixelperformancesummary.db')
)

process.prod = cms.EDAnalyzer("SiPixelPerformanceSummaryReader")

#process.print = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.prod)
#process.ep = cms.EndPath(process.print)


