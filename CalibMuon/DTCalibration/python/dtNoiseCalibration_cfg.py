import FWCore.ParameterSet.Config as cms

process = cms.Process("Calibration")

process.load("CalibMuon.DTCalibration.messageLoggerDebug_cff")
process.MessageLogger.debugModules = ['dtNoiseCalibration']

process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff")
process.GlobalTag.globaltag = ''

process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring()
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.load("CalibMuon.DTCalibration.dt_offlineAnalysis_common_cff")

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    DBParameters = cms.PSet(),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:noise.db'),
    authenticationMethod = cms.untracked.uint32(0),
    toPut = cms.VPSet(
        cms.PSet(
            record = cms.string('DTStatusFlagRcd'),
            tag = cms.string('noise')
        )
    )
)

process.load("CalibMuon.DTCalibration.dtNoiseCalibration_cfi")

#process.p = cms.Path(process.muonDTDigis*dtNoiseCalibration)
process.p = cms.Path(process.dtNoiseCalibration)
