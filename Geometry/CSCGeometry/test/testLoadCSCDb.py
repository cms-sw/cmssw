
import FWCore.ParameterSet.Config as cms

process = cms.Process("CSCGeometryWriter")
process.load("CondCore.DBCommon.CondDBCommon_cfi")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cff")

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
        )
process.source = cms.Source("EmptyIOVSource",
                                lastValue = cms.uint64(1),
                                timetype = cms.string('runnumber'),
                                firstValue = cms.uint64(1),
                                interval = cms.uint64(1)
                            )

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          DBParameters = cms.PSet(
                                                          messageLevel = cms.untracked.int32(0),
                                                          authenticationPath = cms.untracked.string('.')
                                                           ),
                                          timetype = cms.untracked.string('runnumber'),
                                          BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
                                          connect = cms.string('sqlite_file:myfile.db'),
                                          toPut = cms.VPSet(cms.PSet(record = cms.string('CSCRecoGeometryRcd'),tag = cms.string('XMLFILE_TEST_01')),
                                                            cms.PSet(record = cms.string('CSCRecoDigiParametersRcd'),tag = cms.string('XMLFILE_TEST_02')))
                              )

process.CSCGeometryWriter = cms.EDAnalyzer("CSCRecoIdealDBLoader")

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
            limit = cms.untracked.int32(0)
        ),
        extension = cms.untracked.string('.out'),
        CSC = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        noLineBreaks = cms.untracked.bool(True),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        CSCNumbering = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),    
        threshold = cms.untracked.string('DEBUG'),
        CSCGeometryBuilderFromDDD = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        RadialStripTopology = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    ),
    # For LogDebug/LogTrace output...
    debugModules = cms.untracked.vstring('*'),
    categories = cms.untracked.vstring('CSC', 
        'CSCNumbering', 
        'CSCGeometryBuilderFromDDD',
        'CSCGeometryBuilder', 
        'CSCGeometryParsFromDD', 
        'RadialStripTopology'),
    destinations = cms.untracked.vstring('log', 
        'errors', 
        'debug')
)

process.myprint = cms.OutputModule("AsciiOutputModule")

process.p1 = cms.Path(process.CSCGeometryWriter)
process.ep = cms.EndPath(process.myprint)
