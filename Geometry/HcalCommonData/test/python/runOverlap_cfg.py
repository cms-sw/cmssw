import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")
process.load("SimGeneral.HepPDTESSource.pdt_cfi")

process.load("Geometry.CMSCommonData.hcalOnlyGeometryXML_cfi")

process.load("SimG4Core.Application.g4SimHits_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        G4cerr = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        G4cout = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        enable = cms.untracked.bool(True)
    )
)

process.Timing = cms.Service("Timing")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        generator = cms.untracked.uint32(456789),
        g4SimHits = cms.untracked.uint32(9876),
        VtxSmeared = cms.untracked.uint32(12345)
    ),
    sourceSeed = cms.untracked.uint32(98765)
)

process.source = cms.Source("EmptySource",
    firstRun        = cms.untracked.uint32(1),
    firstEvent      = cms.untracked.uint32(1)
)

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(211),
        MinEta = cms.double(0.5655),
        MaxEta = cms.double(0.5655),
        MinPhi = cms.double(-0.1309),
        MaxPhi = cms.double(-0.1309),
        MinE   = cms.double(9.99),
        MaxE   = cms.double(10.01)
    ),
    Verbosity = cms.untracked.int32(0),
    AddAntiParticle = cms.bool(False)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.p1 = cms.Path(process.generator*process.g4SimHits)
process.g4SimHits.UseMagneticField = False
process.g4SimHits.Physics.DefaultCutValue = 1.
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    Resolution = cms.untracked.int32(1000),
    type       = cms.string('CheckOverlap'),
    NodeNames  = cms.untracked.vstring('CMSE')
))

