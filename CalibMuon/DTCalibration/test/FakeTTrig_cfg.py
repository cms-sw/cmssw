import FWCore.ParameterSet.Config as cms
 
process = cms.Process("fakeDB")
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
process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        FaketTrig = cms.untracked.uint32(563)
    ),
    sourceSeed = cms.untracked.uint32(98765)
)
 
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBSetup,
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:tTrig.db'),
    authenticationMethod = cms.untracked.uint32(0),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('DTTtrigRcd'),
        tag = cms.string('tTrig')
    ))
)
 
process.FaketTrig = cms.EDFilter("FakeTTrig",
    useTofCorrection = cms.untracked.bool(False),
    useWirePropCorrection = cms.untracked.bool(False),
    fakeTTrigPedestal = cms.untracked.double(500.0),
    vPropWire = cms.untracked.double(24.4),
    smearing = cms.untracked.double(5.0)
)
 
process.p = cms.Path(process.FaketTrig)
 
