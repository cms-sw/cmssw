import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.source = cms.Source("EmptySource")

process.generator = cms.EDFilter("Pythia8PtAndDxyGun",

    maxEventsToPrint = cms.untracked.int32(1),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    pythiaHepMCVerbosity = cms.untracked.bool(True),

    PGunParameters = cms.PSet(
        ParticleID = cms.vint32(15),
        AddAntiParticle = cms.bool(True),
        MinPt = cms.double(15.0),
        MaxPt = cms.double(300.0),
        MinEta = cms.double(-2.5),
        MaxEta = cms.double(2.5),
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double(3.14159265359),
        LxyMax = cms.double(850.0),#make sure most tau generated within 3 outermost tracker layers, Gauss distribution
        LzMax = cms.double(2100.0),#make sure most taus generated within 3 outemost tracker layers, Gauss distribution
        ConeRadius = cms.double(1000.0),
        ConeH = cms.double(3000.0),
        DistanceToAPEX = cms.double(850.0),
        dxyMin = cms.double(0.0),
        dxyMax = cms.double(500.0)
    ),

    Verbosity = cms.untracked.int32(0), ## set to 1 (or greater)  for printouts
    psethack = cms.string('displaced taus'),
    firstRun = cms.untracked.uint32(1),

    PythiaParameters = cms.PSet(
        parameterSets = cms.vstring()  
    )
)

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(2)
        ),
        enable = cms.untracked.bool(True)
    )
)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    generator = cms.PSet(
        initialSeed = cms.untracked.uint32(123456789),
        engineName = cms.untracked.string('HepJamesRandom')
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.GENoutput = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('Py8PtAndDxyGun_displacedTaus.root')
)

process.p = cms.Path(process.generator)
process.outpath = cms.EndPath(process.GENoutput)

process.schedule = cms.Schedule(process.p, process.outpath)
