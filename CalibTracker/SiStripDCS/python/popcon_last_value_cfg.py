# Import configurations
import FWCore.ParameterSet.Config as cms

process = cms.Process("test")

process.load("CalibTracker.SiStripDCS.MessLogger_cfi")

process.SiStripConfigDb = cms.Service("SiStripConfigDb",
    ConfDb = cms.untracked.string('username/password@cms_omds_nolb'),
    TNS_ADMIN = cms.untracked.string('.'),
    UsingDb = cms.untracked.bool(True),
    Partitions = cms.untracked.PSet(
        TPDD = cms.untracked.PSet(
            PartitionName = cms.untracked.string('TP_08-AUG-2008_1'),
            ForceVersions = cms.untracked.bool(True),
            FecVersion    = cms.untracked.vuint32(430,2),
            DcuDetIdsVersion = cms.untracked.vuint32(9,0)
        ),
        TMDD = cms.untracked.PSet(
            PartitionName = cms.untracked.string('TM_08-AUG-2008_1'),
            ForceVersions = cms.untracked.bool(True),
            FecVersion    = cms.untracked.vuint32(428,1),
            DcuDetIdsVersion = cms.untracked.vuint32(9,0)
        ),
        TIDD = cms.untracked.PSet(
            PartitionName = cms.untracked.string('TI_08-AUG-2008_1'),
            ForceVersions = cms.untracked.bool(True),
            FecVersion    = cms.untracked.vuint32(427,1),
            DcuDetIdsVersion = cms.untracked.vuint32(9,0)
        ),
        TODD = cms.untracked.PSet(
            PartitionName = cms.untracked.string('TO_08-AUG-2008_1'),
            ForceVersions = cms.untracked.bool(True),
            FecVersion    = cms.untracked.vuint32(415,3),
            DcuDetIdsVersion = cms.untracked.vuint32(9,0)
        ),
        TEPD2 = cms.untracked.PSet(
            PartitionName = cms.untracked.string('TE_27-JUN-2008_2'),
            ForceVersions = cms.untracked.bool(True),
            DcuPsuMapVersion = cms.untracked.vuint32(211, 2)
        ),
        TMPD = cms.untracked.PSet(
            PartitionName = cms.untracked.string('TE_17-JUN-2008_12'),
            ForceVersions = cms.untracked.bool(True),
            DcuPsuMapVersion = cms.untracked.vuint32(163, 1)
        ),
        TEPD1 = cms.untracked.PSet(
            PartitionName = cms.untracked.string('TE_24-JUN-2008_2'),
            ForceVersions = cms.untracked.bool(True),
            DcuPsuMapVersion = cms.untracked.vuint32(204, 1)
        ),
        TEPD4 = cms.untracked.PSet(
            PartitionName = cms.untracked.string('TE_30-JUN-2008_1'),
            ForceVersions = cms.untracked.bool(True),
            DcuPsuMapVersion = cms.untracked.vuint32(229, 1)
        ),
        TEPD3 = cms.untracked.PSet(
            PartitionName = cms.untracked.string('TE_27-JUN-2008_4'),
            ForceVersions = cms.untracked.bool(True),
            DcuPsuMapVersion = cms.untracked.vuint32(214, 1)
        ),
        TPPD = cms.untracked.PSet(
            PartitionName = cms.untracked.string('TE_17-JUN-2008_11'),
            ForceVersions = cms.untracked.bool(True),
            DcuPsuMapVersion = cms.untracked.vuint32(162, 1)
        ),
        TIPD = cms.untracked.PSet(
            PartitionName = cms.untracked.string('TI_17-JUN-2008_2'),
            ForceVersions = cms.untracked.bool(True),
            DcuPsuMapVersion = cms.untracked.vuint32(157, 1)
        ),
        TIPD2 = cms.untracked.PSet(
            PartitionName = cms.untracked.string('TI_18-JUN-2008_1'),
            ForceVersions = cms.untracked.bool(True),
            DcuPsuMapVersion = cms.untracked.vuint32(165, 1)
        ),
        TIPD3 = cms.untracked.PSet(
            PartitionName = cms.untracked.string('TI_18-JUN-2008_10'),
            ForceVersions = cms.untracked.bool(True),
            DcuPsuMapVersion = cms.untracked.vuint32(179, 1)
        ),
        TIPD4 = cms.untracked.PSet(
            PartitionName = cms.untracked.string('TI_20-JUN-2008_1'),
            ForceVersions = cms.untracked.bool(True),
            DcuPsuMapVersion = cms.untracked.vuint32(192, 1)
        ),
        TIPD5 = cms.untracked.PSet(
            PartitionName = cms.untracked.string('TI_27-JUN-2008_1'),
            ForceVersions = cms.untracked.bool(True),
            DcuPsuMapVersion = cms.untracked.vuint32(212, 1)
        ),
        TIPD6 = cms.untracked.PSet(
            PartitionName = cms.untracked.string('TI_27-JUN-2008_3'),
            ForceVersions = cms.untracked.bool(True),
            DcuPsuMapVersion = cms.untracked.vuint32(218, 1)
        ),
        TOPD = cms.untracked.PSet(
            PartitionName = cms.untracked.string('TO_18-JUN-2008_1_TEST_1'),
            ForceVersions = cms.untracked.bool(True),
            DcuPsuMapVersion = cms.untracked.vuint32(177, 1)
        ),
        TOPD2 = cms.untracked.PSet(
            PartitionName = cms.untracked.string('TO_18-JUN-2008_2'),
            ForceVersions = cms.untracked.bool(True),
            DcuPsuMapVersion = cms.untracked.vuint32(178, 1)
        ),
        TOPD3 = cms.untracked.PSet(
            PartitionName = cms.untracked.string('TO_30-JUN-2008_1'),
            ForceVersions = cms.untracked.bool(True),
            DcuPsuMapVersion = cms.untracked.vuint32(228, 1)
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
process.CondDBCommon.connect = cms.string('oracle://cms_omds_nolb/username')

process.SiStripModuleHVBuilder = cms.Service("SiStripModuleHVBuilder",
    onlineDB = cms.untracked.string('oracle://cms_omds_nolb/username'),
    authPath = cms.untracked.string('.'),
# Format for date/time vector:  year, month, day, hour, minute, second, nanosecond                              
    Tmin = cms.untracked.vint32(2008, 10, 13, 1, 0, 0, 0),
    Tmax = cms.untracked.vint32(2008, 10, 13, 12, 0, 0, 0),
# Do NOT change this unless you know what you are doing!
    TSetMin = cms.untracked.vint32(2007, 11, 26, 0, 0, 0, 0),                                             
# queryType can be either STATUSCHANGE or LASTVALUE                              
    queryType = cms.untracked.string('LASTVALUE'),
# if reading lastValue from file put insert file name here                              
    lastValueFile = cms.untracked.string(''),
# flag to show if you are reading from file for lastValue or not                              
    lastValueFromFile = cms.untracked.bool(False),
#
    debugModeOn = cms.untracked.bool(False)
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
        record = cms.string('SiStripDetVOffRcd'),
        tag = cms.string('SiStripDetVOff_Fake_31X')
    )),
    logconnect = cms.untracked.string('sqlite_file:logfile.db')
)

process.siStripPopConModuleHV = cms.EDAnalyzer("SiStripPopConModuleHV",
    record = cms.string('SiStripDetVOffRcd'),
    loggingOn= cms.untracked.bool(True),
    SinceAppendMode=cms.bool(True),
    Source = cms.PSet(
        name = cms.untracked.string('default')
    )                                        
)

process.p = cms.Path(process.siStripPopConModuleHV)
    
