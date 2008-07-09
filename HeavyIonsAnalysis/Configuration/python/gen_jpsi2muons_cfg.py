import FWCore.ParameterSet.Config as cms

process = cms.Process("GEN")
process.load("GeneratorInterface.Pythia6Interface.pythiaDefault_cff")

process.source = cms.Source("PythiaSource",
    Ptmax = cms.untracked.double(100.0),
    pythiaPylistVerbosity = cms.untracked.int32(0),
    ymax = cms.untracked.double(10.0),
    ParticleID = cms.untracked.int32(443),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    yBinning = cms.untracked.int32(500),
    DoubleParticle = cms.untracked.bool(False),
    Ptmin = cms.untracked.double(0.0),
    kinematicsFile = cms.untracked.string('HeavyIonsAnalysis/Configuration/data/jpsipbpb.root'),
    ymin = cms.untracked.double(-10.0),
    ptBinning = cms.untracked.int32(100000),
    maxEventsToPrint = cms.untracked.int32(0),
    PythiaParameters = cms.PSet(
        process.pythiaDefaultBlock,
        myParameters = cms.vstring('BRAT(858) = 0 ! switch off', 
            'BRAT(859) = 1 ! switch on', 
            'BRAT(860) = 0 ! switch off', 
            'MDME(858,1) = 0 ! switch off', 
            'MDME(859,1) = 1 ! switch on', 
            'MDME(860,1) = 0 ! switch off'),
        parameterSets = cms.vstring('pythiaDefault', 
            'myParameters')
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100000)
)
process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    theSource = cms.PSet(
        initialSeed = cms.untracked.uint32(123456789)
    )
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('gen_jpsi_dimuon.root')
)

process.p = cms.EndPath(process.out)
process.PythiaSource.pythiaPylistVerbosity = 0
process.PythiaSource.maxEventsToPrint = 10
process.PythiaSource.pythiaHepMCVerbosity = False


