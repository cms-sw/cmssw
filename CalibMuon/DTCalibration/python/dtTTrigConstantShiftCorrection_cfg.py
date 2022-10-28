import FWCore.ParameterSet.Config as cms

process = cms.Process("DTTTrigCorrection")

process.load("CalibMuon.DTCalibration.messageLoggerDebug_cff")
process.MessageLogger.debugModules = cms.untracked.vstring('dtTTrigConstantShiftCorrection')

process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag=autoCond['run3_data']

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
        record = cms.string('DTTtrigRcd'),
        tag = cms.string('ttrig')
    ))
)
process.PoolDBOutputService.connect = cms.string('sqlite_file:ttrig.db')

process.load("CalibMuon.DTCalibration.dtTTrigConstantShiftCorrection_cfi")
process.dtTTrigConstantShiftCorrection.correctionAlgoConfig.dbLabel = ''
process.dtTTrigConstantShiftCorrection.correctionAlgoConfig.value = 0.
process.dtTTrigConstantShiftCorrection.correctionAlgoConfig.calibChamber = 'All'

process.p = cms.Path(process.dtTTrigConstantShiftCorrection)
