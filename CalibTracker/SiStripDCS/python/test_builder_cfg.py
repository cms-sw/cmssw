import FWCore.ParameterSet.Config as cms

process = cms.Process("testbuilder")

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

process.load("IORawData.SiStripInputSources.EmptySource_cff")
process.maxEvents.input = 2


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

process.test = cms.EDAnalyzer("testbuilding")

process.p = cms.Path(process.test)
