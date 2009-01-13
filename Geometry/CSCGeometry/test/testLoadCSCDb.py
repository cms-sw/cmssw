
import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")
process.load("CondCore.DBCommon.CondDBCommon_cfi")

#process.load("DetectorDescription.OfflineDBLoader.test.cmsIdealGeometryForWrite_cfi")
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
                                          connect = cms.string('sqlite_file:test.db'),
                                          #                                              process.CondDBCommon,
                                          toPut = cms.VPSet(cms.PSet(
                                                             record = cms.string('CSCRecoGeometryRcd'),
                                                                tag = cms.string('RecoIdealGeometry')
                                                             ),cms.PSet(
                                                             record = cms.string('CSCRecoDigiParametersRcd'),
                                                                tag = cms.string('CSCRecoDigiParameters')
                                                             )
                                                            )
                              )
process.load = cms.EDFilter("CSCRecoIdealDBLoader")

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
        'RadialStripTopology'),
    destinations = cms.untracked.vstring('log', 
        'errors', 
        'debug')
)

process.myprint = cms.OutputModule("AsciiOutputModule")

process.p1 = cms.Path(process.load)
process.ep = cms.EndPath(process.myprint)
process.CondDBCommon.connect = 'sqlite_file:test.db'
process.CondDBCommon.DBParameters.messageLevel = 0
process.CondDBCommon.DBParameters.authenticationPath = './'
