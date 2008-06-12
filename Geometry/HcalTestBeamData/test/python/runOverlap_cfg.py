import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")
process.load("SimGeneral.HepPDTESSource.pdt_cfi")

process.load("Geometry.HcalTestBeamData.test.2007.TB2007GeometryXML_cfi")

process.load("SimG4Core.Application.g4SimHits_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        G4cout = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        G4cerr = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        HCalGeom = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    ),
    categories = cms.untracked.vstring('G4cout', 
        'G4cerr', 
        'HCalGeom'),
    destinations = cms.untracked.vstring('cout')
)

process.Timing = cms.Service("Timing")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        g4SimHits = cms.untracked.uint32(9876),
        VtxSmeared = cms.untracked.uint32(12345)
    ),
    sourceSeed = cms.untracked.uint32(98765)
)

process.common_beam_direction_parameters = cms.PSet(
    MaxEta = cms.untracked.double(0.5655),
    MaxPhi = cms.untracked.double(-0.1309),
    MinEta = cms.untracked.double(0.5655),
    MinPhi = cms.untracked.double(-0.1309),
    BeamPosition = cms.untracked.double(-800.0)
)
process.source = cms.Source("FlatRandomEGunSource",
    PGunParameters = cms.untracked.PSet(
        process.common_beam_direction_parameters,
        MaxE = cms.untracked.double(10.01),
        MinE = cms.untracked.double(9.99),
        PartID = cms.untracked.vint32(211)
    ),
    Verbosity = cms.untracked.int32(0)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.common_heavy_suppression1 = cms.PSet(
    NeutronThreshold = cms.double(30.0),
    ProtonThreshold = cms.double(30.0),
    IonThreshold = cms.double(30.0)
)
process.p1 = cms.Path(process.g4SimHits)
process.g4SimHits.UseMagneticField = False
process.g4SimHits.Physics.DefaultCutValue = 1.
process.g4SimHits.Generator.HepMCProductLabel = 'source'
process.g4SimHits.StackingAction = cms.PSet(
    process.common_heavy_suppression1,
    TrackNeutrino = cms.bool(False),
    KillHeavy = cms.bool(False),
    SavePrimaryDecayProductsAndConversions = cms.untracked.bool(False)
)
process.g4SimHits.CaloSD = cms.PSet(
    process.common_beam_direction_parameters,
    process.common_heavy_suppression1,
    SuppressHeavy = cms.bool(False),
    DetailedTiming = cms.untracked.bool(False),
    Verbosity = cms.untracked.int32(0),
    CheckHits = cms.untracked.int32(25),
    CorrectTOFBeam = cms.untracked.bool(False),
    UseMap = cms.untracked.bool(True),
    EminTrack = cms.double(1.0)
)
process.g4SimHits.HCalSD.UseShowerLibrary = False
process.g4SimHits.HCalSD.TestNumberingScheme = True
process.g4SimHits.HCalSD.UseHF = False
process.g4SimHits.HCalSD.ForTBH2 = True
process.g4SimHits.CaloTrkProcessing.TestBeam = True
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    Resolution = cms.untracked.int32(1000),
    type = cms.string('CheckOverlap'),
    NodeName = cms.untracked.string('')
))

