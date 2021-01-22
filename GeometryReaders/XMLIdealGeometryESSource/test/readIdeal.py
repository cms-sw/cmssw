import FWCore.ParameterSet.Config as cms

process = cms.Process("CompareGeometryTest")
process.load('Configuration.Geometry.GeometryIdeal2015_cff')

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

process.pAStd = cms.EDAnalyzer("PerfectGeometryAnalyzer",
                               dumpPosInfo = cms.untracked.bool(False),
                               label = cms.untracked.string(''),
                               isMagField = cms.untracked.bool(False),
                               dumpSpecs = cms.untracked.bool(False),
                               dumpGeoHistory = cms.untracked.bool(False),
                               outFileName = cms.untracked.string('STD'),
                               numNodesToDump = cms.untracked.uint32(0),
                               fromDB = cms.untracked.bool(False),
                               ddRootNodeName = cms.untracked.string('cms:OCMS')
                               )

process.BigXMLWriter = cms.EDAnalyzer("OutputDDToDDL",
                              rotNumSeed = cms.int32(0),
                              fileName = cms.untracked.string('fred.xml')
                              )

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    debugModules = cms.untracked.vstring('*'),
    files = cms.untracked.PSet(
        readIdealdebug = cms.untracked.PSet(
            DEBUG = cms.untracked.PSet(
                limit = cms.untracked.int32(-1)
            ),
            INFO = cms.untracked.PSet(
                limit = cms.untracked.int32(-1)
            ),
            extension = cms.untracked.string('.out'),
            noLineBreaks = cms.untracked.bool(True),
            threshold = cms.untracked.string('DEBUG')
        ),
        readIdealerrors = cms.untracked.PSet(
            extension = cms.untracked.string('.out'),
            threshold = cms.untracked.string('ERROR')
        )
    )
)

process.Timing = cms.Service("Timing")

process.p1 = cms.Path(process.pAStd)

process.e1 = cms.EndPath(process.myprint)
