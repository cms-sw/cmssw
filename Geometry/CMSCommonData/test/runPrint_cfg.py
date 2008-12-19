import FWCore.ParameterSet.Config as cms

process = cms.Process("PrintGeom")

process.load("Validation.CheckOverlap.testGeometry_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout'),
    categories = cms.untracked.vstring('FwkJob'),
    fwkJobReports = cms.untracked.vstring('FrameworkJobReport'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    FrameworkJobReport = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        FwkJob = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(0)
)

process.load("SimGeneral.HepPDTESSource.pdt_cfi")

process.source = cms.Source("FlatRandomPtGunSource",
    PGunParameters = cms.untracked.PSet(
        PartID = cms.untracked.vint32(13),
        MinEta = cms.untracked.double(-2.5),
        MaxEta = cms.untracked.double(2.5),
        MinPhi = cms.untracked.double(-3.14159265359),
        MaxPhi = cms.untracked.double(3.14159265359),
        MinPt  = cms.untracked.double(9.99),
        MaxPt  = cms.untracked.double(10.01)
    ),
    Verbosity = cms.untracked.int32(0),
    AddAntiParticle = cms.untracked.bool(False),
    firstRun = cms.untracked.uint32(1)
)


process.EnableFloatingPointExceptions = cms.Service("EnableFloatingPointExceptions",
    enableDivByZeroEx = cms.untracked.bool(False),
    enableInvalidEx   = cms.untracked.bool(True),
    enableOverFlowEx  = cms.untracked.bool(False),
    enableUnderFlowEx = cms.untracked.bool(False)
)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        g4SimHits = cms.untracked.uint32(9876),
        VtxSmeared = cms.untracked.uint32(98765432)
    ),
    sourceSeed = cms.untracked.uint32(123456789)
)

process.load("SimG4Core.Application.g4SimHits_cfi")

process.p1 = cms.Path(process.g4SimHits)

process.g4SimHits.Physics.type            = 'SimG4Core/Physics/DummyPhysics'
process.g4SimHits.UseMagneticField        = False
process.g4SimHits.Physics.DummyEMPhysics  = True
process.g4SimHits.Physics.DefaultCutValue = 10. 
process.g4SimHits.Generator.HepMCProductLabel = 'source'
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
	DumpSummary    = cms.untracked.bool(True),
	DumpLVTree     = cms.untracked.bool(True),
	DumpMaterial   = cms.untracked.bool(False),
	DumpLVList     = cms.untracked.bool(True),
	DumpLV         = cms.untracked.bool(True),
	DumpSolid      = cms.untracked.bool(True),
	DumpAttributes = cms.untracked.bool(False),
	DumpPV         = cms.untracked.bool(True),
	DumpRotation   = cms.untracked.bool(False),
	DumpReplica    = cms.untracked.bool(False),
	DumpTouch      = cms.untracked.bool(False),
	DumpSense      = cms.untracked.bool(False),
	Name           = cms.untracked.string('*'),
	Names          = cms.untracked.vstring(),
	type           = cms.string('PrintGeomInfoAction')
))
