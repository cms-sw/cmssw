import FWCore.ParameterSet.Config as cms

process = cms.Process("DTTTrigCorrection")

process.load("CalibMuon.DTCalibration.messageLoggerDebug_cff")
process.MessageLogger.debugModules = cms.untracked.vstring('dtTTrigResidualCorrection')

process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
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
        record = cms.string('DTTtrigRcd'),
        tag = cms.string('ttrig')
    ))
)
process.PoolDBOutputService.connect = cms.string('sqlite_file:ttrig.db')

process.load("CalibMuon.DTCalibration.dtTTrigResidualCorrection_cfi")
process.dtTTrigResidualCorrection.correctionAlgoConfig.residualsRootFile = 'DTkFactValidation.root'

process.p = cms.Path(process.dtTTrigResidualCorrection)
