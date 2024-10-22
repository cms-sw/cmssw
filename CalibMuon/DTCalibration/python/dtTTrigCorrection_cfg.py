import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

process = cms.Process("DTTTrigCorrection",eras.Run3)

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

process.load("CalibMuon.DTCalibration.dtTTrigCorrection_cfi")

process.p = cms.Path(process.dtTTrigCorrection)
