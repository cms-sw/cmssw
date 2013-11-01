import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")

#process.source = cms.Source("EmptySource")
process.source = cms.Source("LHESource",
    fileNames = cms.untracked.vstring('file:powheg-hvq.lhe')
)
#    fileNames = cms.untracked.vstring('file:powheg-Zee.lhe')

process.generator = cms.EDFilter("Pythia8HadronizerFilter",
    maxEventsToPrint = cms.untracked.int32(1),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    comEnergy = cms.double(7000.),
    emissionVeto = cms.untracked.PSet(),
    #EV_CheckHard = cms.bool(True),
    PythiaParameters = cms.PSet(
        pythia8_example07 = cms.vstring('SpaceShower:pTmaxMatch = 2',
                                        'TimeShower:pTmaxMatch  = 2'),
        parameterSets = cms.vstring('pythia8_example07')
    )
)

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    destinations = cms.untracked.vstring('cout')
)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        generator = cms.untracked.uint32(123456),
        g4SimHits = cms.untracked.uint32(123456788),
        VtxSmeared = cms.untracked.uint32(123456789)
    ),
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

process.GEN = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('pythia8ex7.root')
)

process.p = cms.Path(process.generator)
process.outpath = cms.EndPath(process.GEN)

process.schedule = cms.Schedule(process.p, process.outpath)
