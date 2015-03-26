import FWCore.ParameterSet.Config as cms

process = cms.Process("CompareGeometryTest")

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
        )

process.source = cms.Source("EmptyIOVSource",
                            lastValue = cms.uint64(1),
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )

process.myprint = cms.OutputModule("AsciiOutputModule")

process.fred = cms.ESSource("XMLIdealGeometryESSource",
                            geomXMLFiles = cms.vstring('GeometryReaders/XMLIdealGeometryESSource/test/fred.xml'),
                            rootNodeName = cms.string('cms:OCMS')
                            )

process.pABF = cms.EDAnalyzer("PerfectGeometryAnalyzer",
                              ddRootNodeName = cms.untracked.string("cms:OCMS"),
                              dumpPosInfo = cms.untracked.bool(True),
                              label = cms.untracked.string("fred"),
                              isMagField = cms.untracked.bool(False),
                              dumpSpecs = cms.untracked.bool(True),
                              dumpGeoHistory = cms.untracked.bool(True),
                              outFileName = cms.untracked.string("BDB"),
                              numNodesToDump = cms.untracked.uint32(0)
                              )

process.MessageLogger = cms.Service("MessageLogger",
    errors = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR'),
        extension = cms.untracked.string('.out')
    ),
    # No constraint on log content...equivalent to threshold INFO
    # 0 means none, -1 means all (?)
    log = cms.untracked.PSet(
        extension = cms.untracked.string('.out')
    ),
    debug = cms.untracked.PSet(
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        extension = cms.untracked.string('.out'),

        noLineBreaks = cms.untracked.bool(True),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        threshold = cms.untracked.string('DEBUG'),
    ),
    # For LogDebug/LogTrace output...
    debugModules = cms.untracked.vstring('*'),
    categories = cms.untracked.vstring('*'),
    destinations = cms.untracked.vstring('log', 
        'errors', 
        'debug')
)

process.Timing = cms.Service("Timing")

process.p1 = cms.Path(process.pABF)
process.e1 = cms.EndPath(process.myprint)
