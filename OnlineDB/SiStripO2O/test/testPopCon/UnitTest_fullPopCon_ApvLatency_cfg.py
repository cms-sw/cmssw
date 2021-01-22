import FWCore.ParameterSet.Config as cms

process = cms.Process("o2o")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('DEBUG')
    ),
    debugModules = cms.untracked.vstring('*')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)


process.load("OnlineDB.SiStripConfigDb.SiStripConfigDb_cfi")
process.SiStripConfigDb.UsingDb = True
process.SiStripConfigDb.ConfDb = ''
process.SiStripConfigDb.Partitions = cms.untracked.PSet(
    PartTIBD = cms.untracked.PSet(
        ForceCurrentState = cms.untracked.bool(False),
        ForceVersions = cms.untracked.bool(True),
        PartitionName = cms.untracked.string('TI_13-JUN-2009_1'),
        # RunNumber = cms.untracked.uint32(105505),
        # RunNumber = cms.untracked.uint32(108838),
        RunNumber = cms.untracked.uint32(111799),
        CablingVersion = cms.untracked.vuint32(72, 0),
        FecVersion = cms.untracked.vuint32(555, 0),
        FedVersion = cms.untracked.vuint32(723, 0),
        DcuDetIdsVersion = cms.untracked.vuint32(9, 0),
        DcuPsuMapVersion = cms.untracked.vuint32(265,1)
        )
    )
process.SiStripConfigDb.TNS_ADMIN = ''

process.load("OnlineDB.SiStripO2O.SiStripO2OCalibrationFactors_cfi")
process.SiStripCondObjBuilderFromDb = cms.Service("SiStripCondObjBuilderFromDb",
    process.SiStripO2OCalibrationFactors
)
process.SiStripCondObjBuilderFromDb.SiStripDetInfoFile = cms.FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat")
process.SiStripCondObjBuilderFromDb.UseFEC = True

process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = 'sqlite_file:dbfile.db'
process.CondDBCommon.DBParameters.messageLevel = 4
process.CondDBCommon.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    logconnect = cms.untracked.string('sqlite_file:log.db'),
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiStripLatencyRcd'),
        tag = cms.string('SiStripApvLatency_test')
    ))
)

process.CommonSiStripPopConParams = cms.PSet(
    SinceAppendMode = cms.bool(True),
    Source = cms.PSet(
        since = cms.untracked.uint32(1),
        name = cms.untracked.string('default'),
        debug = cms.untracked.bool(True)
    ),
    loggingOn = cms.untracked.bool(True)
)

process.siStripPopConApvLatency = cms.EDAnalyzer("SiStripPopConApvLatency",
    process.CommonSiStripPopConParams,
    record = cms.string('SiStripLatencyRcd')
)
process.siStripPopConApvLatency.Source.name = 'siStripPopConApvLatency'


process.pped = cms.Path(process.siStripPopConApvLatency)


