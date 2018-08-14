import FWCore.ParameterSet.Config as cms

process = cms.Process("DTVDriftWriter")

process.load("CalibMuon.DTCalibration.messageLoggerDebug_cff")
process.MessageLogger.debugModules = cms.untracked.vstring('dtVDriftSegmentWriter')

process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff")
process.GlobalTag.globaltag = ''

process.load("CondCore.CondDB.CondDB_cfi")

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDB,
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('DTMtimeRcd'),
        tag = cms.string('vDrift')
    ))
)
process.PoolDBOutputService.connect = cms.string('sqlite_file:vDrift.db')

process.load("CalibMuon.DTCalibration.dtVDriftSegmentWriter_cfi")

process.p = cms.Path(process.dtVDriftSegmentWriter)
