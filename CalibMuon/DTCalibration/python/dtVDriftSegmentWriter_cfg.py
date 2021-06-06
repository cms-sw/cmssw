import FWCore.ParameterSet.Config as cms

process = cms.Process("DTVDriftWriter")

### Set to true to switch to writing constants in the new DB format.
NEWDBFORMAT = False 
###

process.load("CalibMuon.DTCalibration.messageLoggerDebug_cff")
process.MessageLogger.debugModules = cms.untracked.vstring('dtVDriftSegmentWriter')

process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = ''

process.load("CondCore.CondDB.CondDB_cfi")

process.load("CalibMuon.DTCalibration.dtVDriftSegmentWriter_cfi")

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

RECORD = 'DTMtimeRcd'
if NEWDBFORMAT :
    RECORD = 'DTRecoConditionsVdriftRcd'
    process.dtVDriftSegmentWriter.writeLegacyVDriftDB = False
    # The following needs to be set as well if calibration should start use
    # constants written in the new format as a starting point.
    # process.dtVDriftSegmentWriter.vDriftAlgoConfig.readLegacyVDriftDB = False

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDB,
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string(RECORD),
        tag = cms.string('vDrift')
    ))
)
process.PoolDBOutputService.connect = cms.string('sqlite_file:vDrift.db')

process.p = cms.Path(process.dtVDriftSegmentWriter)
