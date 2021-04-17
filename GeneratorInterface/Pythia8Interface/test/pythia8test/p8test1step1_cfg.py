import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")

process.source = cms.Source("EmptySource")

process.generator = cms.EDFilter("Pythia8GeneratorFilter",
    maxEventsToPrint = cms.untracked.int32(1),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    comEnergy = cms.double(14000.),
    PythiaParameters = cms.PSet(
        pythia8_test1 = cms.vstring('WeakBosonAndParton:qqbar2gmZg = on',
                                        'WeakBosonAndParton:qg2gmZq = on',
                                        'PhaseSpace:pTHatMin = 20.',
                                        '23:onMode = off',
                                        '23:onIfAny = 11 13 15'),
        parameterSets = cms.vstring('pythia8_test1')
    )
)

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        enable = cms.untracked.bool(True)
    )
)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    generator = cms.PSet(
        initialSeed = cms.untracked.uint32(123456789),
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.GEN = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('p8test1step1.root')
)

process.p = cms.Path(process.generator)
process.outpath = cms.EndPath(process.GEN)

process.schedule = cms.Schedule(process.p, process.outpath)

