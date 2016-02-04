 
import FWCore.ParameterSet.Config as cms
 
process = cms.Process("ProdFakeDB")
process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")
 
process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.DTGeometryESModule.applyAlignment = False
 
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

from CalibTracker.Configuration.Common.PoolDBESSource_cfi import poolDBESSource
poolDBESSource.connect = "frontier://FrontierDev/CMS_COND_ALIGNMENT"
poolDBESSource.toGet = cms.VPSet(cms.PSet(
        record = cms.string('GlobalPositionRcd'),
        tag = cms.string('IdealGeometry')
    ))

process.load("CondCore.DBCommon.CondDBSetup_cfi")
 
process.source = cms.Source("EmptySource")
 
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBSetup,
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:vDrift.db'),
    authenticationMethod = cms.untracked.uint32(0),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('DTMtimeRcd'),
        tag = cms.string('vDrift')
    ))
)
 
process.prodFakeDb = cms.EDAnalyzer("ProduceFakeDB",
    dbToProduce = cms.untracked.string('VDriftDB'),
    hitResolution = cms.untracked.double(0.02),
    mapToProduce = cms.untracked.string('VDriftDB'),
    vdrift = cms.untracked.double(0.00543)
)
 
process.p = cms.Path(process.prodFakeDb)
 
