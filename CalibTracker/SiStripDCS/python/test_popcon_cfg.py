# Import configurations
import FWCore.ParameterSet.Config as cms

process = cms.Process("test")

process.load("DQM.SiStripCommon.MessageLogger_cfi")

process.SiStripConfigDb = cms.Service("SiStripConfigDb",
    ConfDb = cms.untracked.string(''),
    TNS_ADMIN = cms.untracked.string(''),
    UsingDb = cms.untracked.bool(True),
    Partitions = cms.untracked.PSet(
        DCUDETID = cms.untracked.PSet(
            PartitionName = cms.untracked.string(''),
            ForceCurrentState = cms.untracked.bool(True)
        ),
        DCUPSU = cms.untracked.PSet(
            PartitionName = cms.untracked.string(''),
            ForceVersions = cms.untracked.bool(True),
            DcuPsuMapVersion = cms.untracked.vuint32(0, 0)
        )
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = cms.string('')

process.SiStripModuleHVBuilder = cms.Service("SiStripModuleHVBuilder",
    onlineDB = cms.untracked.string(''),
    authPath = cms.untracked.string(''),
# Format for date/time vector:  year, month, day, hour, minute, second, nanosecond
    Tmin = cms.untracked.vint32(0, 0, 0, 0, 0, 0, 0),
    Tmax = cms.untracked.vint32(0, 0, 0, 0, 0, 0, 0),
# queryType can be either STATUSCHANGE or LASTVALUE                              
    queryType = cms.untracked.string('STATUSCHANGE'),
# if reading lastValue from file put insert file name here                              
    lastValueFile = cms.untracked.string(''),
# flag to show if you are reading from file for lastValue or not                              
    lastValueFromFile = cms.untracked.bool(False)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(2),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('timestamp'),
    connect = cms.string('sqlite_file:dbfile.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiStripModuleHVRcd'),
        tag = cms.string('SiStripModuleHV_Fake_30X')
    )),
    logconnect = cms.untracked.string('sqlite_file:logfile.db')
)

process.siStripPopConModuleHV = cms.EDAnalyzer("SiStripPopConModuleHV",
    record = cms.string('SiStripModuleHVRcd'),
    loggingOn= cms.untracked.bool(True),
    SinceAppendMode=cms.bool(True),
    Source = cms.PSet(
        name = cms.untracked.string('default')
    )                                        
)

process.p = cms.Path(process.siStripPopConModuleHV)
    
