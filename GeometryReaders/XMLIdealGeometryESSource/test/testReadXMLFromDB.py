import FWCore.ParameterSet.Config as cms

process = cms.Process("DBGeometryTest")
process.load('FWCore.MessageLogger.MessageLogger_cfi')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('CondCore.DBCommon.CondDBSetup_cfi')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['mc']

process.source = cms.Source("EmptySource")
process.XMLFromDBSource.label=''

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.GlobalTag.toGet = cms.VPSet(cms.PSet(record = cms.string('GeometryFileRcd'),
                                             tag = cms.string('XMLFILE_Geometry_TagXX_Extended2015_mc'),
                                             connect = cms.untracked.string('sqlite_file:./myfile.db')
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
                                    readDBerrors = cms.untracked.PSet( threshold = cms.untracked.string('ERROR'),
                                                                          extension = cms.untracked.string('.out')
                                                                          ),
                                    readDBdebug = cms.untracked.PSet( INFO = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
                                                                         extension = cms.untracked.string('.out'),
                                                                         noLineBreaks = cms.untracked.bool(True),
                                                                         DEBUG = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
                                                                         threshold = cms.untracked.string('DEBUG'),
                                                                         ),
                                    # For LogDebug/LogTrace output...
                                    debugModules = cms.untracked.vstring('*'),
                                    categories = cms.untracked.vstring('*'),
                                    destinations = cms.untracked.vstring('readDBerrors','readDBdebug')
                                    )

process.Timing = cms.Service("Timing")

process.p1 = cms.Path(process.pDB)
process.e1 = cms.EndPath(process.myprint)
