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
      PartTIBD= cms.untracked.PSet(
                PartitionName = cms.untracked.string("TI_13-JUN-2009_1"),
                ForceCurrentState = cms.untracked.bool(False),
                ForceVersions = cms.untracked.bool(True), 
                CablingVersion = cms.untracked.vuint32(72,0),
                FecVersion = cms.untracked.vuint32(568,0),
                FedVersion = cms.untracked.vuint32(747,0),
                DcuDetIdsVersion = cms.untracked.vuint32(9,0),
                MaskVersion = cms.untracked.vuint32(85,0),
                DcuPsuMapVersion = cms.untracked.vuint32(265,1)
                
        )
    )
process.SiStripConfigDb.TNS_ADMIN = ''

process.load("OnlineDB.SiStripO2O.SiStripO2OCalibrationFactors_cfi")
process.SiStripCondObjBuilderFromDb = cms.Service("SiStripCondObjBuilderFromDb",
    process.SiStripO2OCalibrationFactors
)
process.SiStripCondObjBuilderFromDb.SiStripDetInfoFile = cms.FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat")
process.SiStripCondObjBuilderFromDb.UseFED = True

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
        record = cms.string('SiStripPedestalsRcd'),
        tag = cms.string('SiStripPedestals_test')
    ), 
        cms.PSet(
            record = cms.string('SiStripNoisesRcd'),
            tag = cms.string('SiStripNoise_test')
        ), 
        cms.PSet(
            record = cms.string('SiStripThresholdRcd'),
            tag = cms.string('SiStripThreshold_test')
        ), 
        cms.PSet(
            record = cms.string('SiStripBadStrip'),
            tag = cms.string('SiStripBadChannel_test')
        ), 
        cms.PSet(
            record = cms.string('SiStripFedCablingRcd'),
            tag = cms.string('SiStripFedCabling_test')
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

process.siStripPopConNoise = cms.EDAnalyzer("SiStripPopConNoise",
    process.CommonSiStripPopConParams,
    record = cms.string('SiStripNoisesRcd')
)
process.siStripPopConNoise.Source.name = 'siStripPopConNoise'

process.siStripPopConPedestals = cms.EDAnalyzer("SiStripPopConPedestals",
    process.CommonSiStripPopConParams,
    record = cms.string('SiStripPedestalsRcd')
)
process.siStripPopConPedestals.Source.name = 'siStripPopConPedestals'

process.siStripPopConThreshold = cms.EDAnalyzer("SiStripPopConThreshold",
    process.CommonSiStripPopConParams,
    record = cms.string('SiStripThresholdRcd')
)
process.siStripPopConThreshold.Source.name = 'siStripPopConThreshold'

process.siStripPopConFedCabling = cms.EDAnalyzer("SiStripPopConFedCabling",
    process.CommonSiStripPopConParams,
    record = cms.string('SiStripFedCablingRcd')
)
process.siStripPopConFedCabling.Source.name = 'siStripPopConFedCabling'

process.siStripPopConBadStrip = cms.EDAnalyzer("SiStripPopConBadStrip",
    process.CommonSiStripPopConParams,
    record = cms.string('SiStripBadStrip')
)
process.siStripPopConBadStrip.Source.name = 'siStripPopConBadStrip'


process.pped = cms.Path(process.siStripPopConFedCabling+process.siStripPopConNoise+process.siStripPopConPedestals+process.siStripPopConThreshold+process.siStripPopConBadStrip)




