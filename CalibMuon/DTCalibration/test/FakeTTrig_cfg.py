import FWCore.ParameterSet.Config as cms
 
process = cms.Process("fakeDB")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.DTGeometryESModule.applyAlignment = False

process.load("CondCore.DBCommon.CondDBSetup_cfi")
 
process.source = cms.Source("EmptySource")
 
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.calibDB = cms.ESSource("PoolDBESSource",
    process.CondDBSetup,
    authenticationMethod = cms.untracked.uint32(0),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('DTTtrigRcd'),
        tag = cms.string('ttrig')
    )),
    connect = cms.string('sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/COMM09/ttrig/ttrig_ResidCorr_112281.db')
)

#process.DTMapping = cms.ESSource("PoolDBESSource",
#         DBParameters = cms.PSet(
#         messageLevel = cms.untracked.int32(0),
#         authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
#         ),
#         siteLocalConfig = cms.untracked.bool(False),
#         toGet = cms.VPSet(cms.PSet(
#         record = cms.string('DTTtrigRcd'),
#         tag = cms.string('DT_tTrig_IDEAL_V01_mc')
#             )),
#         connect = cms.string('frontier://FrontierPrep/CMS_COND_PRESH')
#)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    FaketTrig = cms.PSet(
        initialSeed = cms.untracked.uint32(563)
    )
)
 
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBSetup,
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:ttrig_112281-75.db'),
    authenticationMethod = cms.untracked.uint32(0),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('DTTtrigRcd'),
        tag = cms.string('ttrig')
    ))
)
 
process.FaketTrig = cms.EDAnalyzer("FakeTTrig",
    useTofCorrection = cms.untracked.bool(False),
    useWirePropCorrection = cms.untracked.bool(False),
    dbLabel = cms.untracked.string(''),
    vPropWire = cms.untracked.double(24.4),
    readDB = cms.untracked.bool(True),
    fakeTTrigPedestal = cms.untracked.double(500.0),  
    smearing = cms.untracked.double(12.0)
)
 
process.p = cms.Path(process.FaketTrig)
 
