import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")
process.load("SimGeneral.HepPDTESSource.pdt_cfi")

process.load("Geometry.HcalTestBeamData.TB2007TestGeometryXML_cfi")

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
        HCalGeom = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('DEBUG')
    ),
    debugModules = cms.untracked.vstring('*')
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

process.common_beam_direction_parameters = cms.PSet(
    MaxEta       = cms.double(0.5655),
    MinEta       = cms.double(0.5655),
    MaxPhi       = cms.double(-0.1309),
    MinPhi       = cms.double(-0.1309),
    BeamPosition = cms.double(-800.0)
)

process.source = cms.Source("EmptySource")

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        process.common_beam_direction_parameters,
        MinE   = cms.double(9.99),
        MaxE   = cms.double(10.01),
        PartID = cms.vint32(211)
    ),
    Verbosity       = cms.untracked.int32(0),
    AddAntiParticle = cms.bool(False),
    firstRun        = cms.untracked.uint32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.common_heavy_suppression1 = cms.PSet(
    NeutronThreshold = cms.double(30.0),
    ProtonThreshold = cms.double(30.0),
    IonThreshold = cms.double(30.0)
)
process.common_maximum_timex = cms.PSet(
    MaxTrackTime  = cms.double(500.0),
    MaxTimeNames  = cms.vstring(),
    MaxTrackTimes = cms.vdouble()
)
process.p1 = cms.Path(process.generator*process.g4SimHits)
process.g4SimHits.UseMagneticField = False
process.g4SimHits.Physics.DefaultCutValue = 1.
process.g4SimHits.Generator.HepMCProductLabel = 'source'
process.g4SimHits.StackingAction = cms.PSet(
    process.common_heavy_suppression1,
    process.common_maximum_timex,
    TrackNeutrino = cms.bool(False),
    KillHeavy = cms.bool(False),
    SaveFirstLevelSecondary = cms.untracked.bool(False),
    SavePrimaryDecayProductsAndConversionsInTracker = cms.untracked.bool(True),
    SavePrimaryDecayProductsAndConversionsInCalo = cms.untracked.bool(False),
    SavePrimaryDecayProductsAndConversionsInMuon = cms.untracked.bool(False)
)
process.g4SimHits.SteppingAction = cms.PSet(
    process.common_maximum_timex,
    KillBeamPipe            = cms.bool(True),
    CriticalEnergyForVacuum = cms.double(2.0),
    CriticalDensity         = cms.double(1e-15),
    EkinNames               = cms.vstring(),
    EkinThresholds          = cms.vdouble(),
    EkinParticles           = cms.vstring(),
    Verbosity = cms.untracked.int32(0)
)
process.g4SimHits.CaloSD = cms.PSet(
    process.common_beam_direction_parameters,
    process.common_heavy_suppression1,
    SuppressHeavy  = cms.bool(False),
    DetailedTiming = cms.untracked.bool(False),
    Verbosity      = cms.untracked.int32(0),
    CheckHits      = cms.untracked.int32(25),
    CorrectTOFBeam = cms.bool(False),
    TmaxHit        = cms.double(1000.0),
    HCNames        = cms.vstring(),
    EminHits       = cms.vdouble(),
    TmaxHits       = cms.vdouble(),
    UseMap         = cms.untracked.bool(True),
    EminTrack      = cms.double(1.0)
)
process.g4SimHits.HCalSD.UseShowerLibrary = False
process.g4SimHits.HCalSD.TestNumberingScheme = True
process.g4SimHits.HCalSD.UseHF = False
process.g4SimHits.HCalSD.ForTBH2 = True
process.g4SimHits.CaloTrkProcessing.TestBeam = True
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    Resolution = cms.untracked.int32(1000),
    type       = cms.string('CheckOverlap'),
    NodeNames  = cms.untracked.vstring('TBHCal')
))

