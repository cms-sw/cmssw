import FWCore.ParameterSet.Config as cms

process = cms.Process("DTTPDeadWriter")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag=autoCond['run3_data']

process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.DTGeometryESModule.applyAlignment = False
process.DTGeometryESModule.fromDDD = False

process.load("CondCore.CondDB.CondDB_cfi")

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

#process.t0DB = cms.ESSource("PoolDBESSource",
#    process.CondDBSetup,
#    toGet = cms.VPSet(cms.PSet(
#        record = cms.string('DTT0Rcd'),
#        tag = cms.string('t0')
#    )),
#)
#process.t0DB.connect = cms.string('sqlite_file:t0.db')
#process.es_prefer_t0DB = cms.ESPrefer('PoolDBESSource','t0DB')

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDB,
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('DTDeadFlagRcd'),
        tag = cms.string('tpDead')
    ))
)
process.PoolDBOutputService.connect = cms.string('sqlite_file:tpDead.db')

process.dtTPDeadWriter = cms.EDAnalyzer("DTTPDeadWriter",
    debug = cms.untracked.bool(True)
)

process.p = cms.Path(process.dtTPDeadWriter)
