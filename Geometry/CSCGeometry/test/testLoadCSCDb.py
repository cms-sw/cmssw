
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
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    debugModules = cms.untracked.vstring('*'),
    files = cms.untracked.PSet(
        debug = cms.untracked.PSet(
            CSC = cms.untracked.PSet(
                limit = cms.untracked.int32(-1)
            ),
            CSCGeometryBuilderFromDDD = cms.untracked.PSet(
                limit = cms.untracked.int32(-1)
            ),
            CSCNumbering = cms.untracked.PSet(
                limit = cms.untracked.int32(-1)
            ),
            DEBUG = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            INFO = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            RadialStripTopology = cms.untracked.PSet(
                limit = cms.untracked.int32(-1)
            ),
            extension = cms.untracked.string('.out'),
            noLineBreaks = cms.untracked.bool(True),
            threshold = cms.untracked.string('DEBUG')
        ),
        errors = cms.untracked.PSet(
            extension = cms.untracked.string('.out'),
            threshold = cms.untracked.string('ERROR')
        ),
        log = cms.untracked.PSet(
            extension = cms.untracked.string('.out')
        )
    )
)

process.myprint = cms.OutputModule("AsciiOutputModule")

process.p1 = cms.Path(process.CSCGeometryWriter)
process.ep = cms.EndPath(process.myprint)
