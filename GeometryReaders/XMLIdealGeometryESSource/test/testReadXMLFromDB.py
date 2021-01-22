import FWCore.ParameterSet.Config as cms

process = cms.Process("DBGeometryTest")
process.load('FWCore.MessageLogger.MessageLogger_cfi')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('CondCore.CondDB.CondDB_cfi')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['mc']

process.source = cms.Source("EmptySource")
process.XMLFromDBSource.label=''

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.GlobalTag.toGet = cms.VPSet(cms.PSet(record = cms.string('GeometryFileRcd'),
                                             tag = cms.string('XMLFILE_Geometry_Extended_TagXX'),
                                             connect = cms.string('sqlite_file:./myfile.db')
                                             )
                                    )

process.myprint = cms.OutputModule("AsciiOutputModule")

XMLFromDBSource = cms.ESProducer("XMLIdealGeometryESProducer",
                                 rootDDName = cms.string('cms:OCMS'),
                                 )

process.pDB = cms.EDAnalyzer("PerfectGeometryAnalyzer",
                             dumpPosInfo = cms.untracked.bool(True),
                             label = cms.untracked.string(''),
                             isMagField = cms.untracked.bool(False),
                             dumpSpecs = cms.untracked.bool(True),
                             dumpGeoHistory = cms.untracked.bool(True),
                             outFileName = cms.untracked.string('LocDB'),
                             numNodesToDump = cms.untracked.uint32(0),
                             fromDB = cms.untracked.bool(True),
                             ddRootNodeName = cms.untracked.string('cms:OCMS')
                             )

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    debugModules = cms.untracked.vstring('*'),
    files = cms.untracked.PSet(
        readDBdebug = cms.untracked.PSet(
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
        readDBerrors = cms.untracked.PSet(
            extension = cms.untracked.string('.out'),
            threshold = cms.untracked.string('ERROR')
        )
    )
)

process.Timing = cms.Service("Timing")

process.p1 = cms.Path(process.pDB)
process.e1 = cms.EndPath(process.myprint)
