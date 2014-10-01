# Left-Right symmetric model WR (m=1500) production with subsequent decay
# through the chain e Ne -> e e jet jet
# Heavy neutrino Ne mass = 600 GeV, others very heavy

import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")

process.source = cms.Source("EmptySource")

process.generator = cms.EDFilter("Pythia8GeneratorFilter",
    maxEventsToPrint = cms.untracked.int32(1),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    comEnergy = cms.double(7000.),
    PythiaParameters = cms.PSet(
        pythia8_example04 = cms.vstring('LeftRightSymmmetry:ffbar2WR = on',
                                        '9900024:m0 = 1500',
                                        '9900012:m0 = 100000',
                                        '9900014:m0 = 100000',
                                        '9900016:m0 = 100000',
                                        '9900012:m0 = 600',
                    '9900024:onMode = off',
                    '9900024:onIfAny = 9900012 9900014 9900016' ),
        parameterSets = cms.vstring('pythia8_example04')
    )
)

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(2)
        )
    ),
    destinations = cms.untracked.vstring('cout')
)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    generator = cms.PSet(
        initialSeed = cms.untracked.uint32(123456789),
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.GEN = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('pythia8ex4.root')
)

process.p = cms.Path(process.generator)
process.outpath = cms.EndPath(process.GEN)

process.schedule = cms.Schedule(process.p, process.outpath)

