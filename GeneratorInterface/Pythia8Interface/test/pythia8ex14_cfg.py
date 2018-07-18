# Using DIRE plugin

import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")

process.source = cms.Source("EmptySource")

process.generator = cms.EDFilter("Pythia8GeneratorFilter",
    maxEventsToPrint = cms.untracked.int32(1),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    comEnergy = cms.double(91.188),
    ElectronPositronInitialState = cms.PSet(),
    DirePlugin = cms.PSet(),
    PythiaParameters = cms.PSet(
        pythia8_example14 = cms.vstring('WeakSingleBoson:ffbar2gmZ = on',
                                        '23:onMode = off',
                                        '23:onIfAny = 1 2 3 4 5',
                                        'PDF:lepton = off',
                                        'SpaceShower:QEDshowerByL = off',
                                        'HadronLevel:all = on',
                                        #NLO corrections
                                        'DireTimes:kernelOrder = 3',
                                        #PS variations
                                        'Variations:doVariations = on',
                                        'Variations:muRfsrDown = 0.25',
                                        'Variations:muRfsrUp   = 4.0',),
        DireTune = cms.vstring(# Tuned hadronization from e+e- data
                               "StringPT:sigma = 0.2952",
                               "StringZ:aLund = 0.9704",
                               "StringZ:bLund = 1.0809",
                               "StringZ:aExtraDiquark = 1.3490",
                               "StringFlav:probStoUD = 0.2046",
                               "StringZ:rFactB = 0.8321",
                               "StringZ:aExtraSQuark = 0.0",
                               "TimeShower:pTmin = 0.9",

                               # Tuned MPI and primordial kT to LHC data (UE in dijets + Drell-Yan pT).
                               "SpaceShower:pTmin = 0.9",
                               "MultipartonInteractions:alphaSvalue = 0.1309",
                               "MultipartonInteractions:pT0Ref = 1.729",
                               "MultipartonInteractions:expPow = 1.769",
                               "ColourReconnection:range = 2.1720",
                               "BeamRemnants:primordialKThard = 2.2873",
                               "BeamRemnants:primordialKTsoft =  0.25",
                               "BeamRemnants:reducedKTatHighY =  0.47",),
        parameterSets = cms.vstring('pythia8_example14', 'DireTune')
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
    input = cms.untracked.int32(100)
)

process.GEN = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('pythia8ex14.root')
)

process.p = cms.Path(process.generator)
process.outpath = cms.EndPath(process.GEN)

process.schedule = cms.Schedule(process.p, process.outpath)
