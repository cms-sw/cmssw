import FWCore.ParameterSet.Config as cms
process = cms.Process("TEST")

process.source =cms.Source("EmptySource")

process.maxEvents.input = 10
process.options.numberOfStreams = 2
process.options.numberOfThreads = 2

process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")
process.RandomNumberGeneratorService.gen1 = process.RandomNumberGeneratorService.generator.clone()
process.RandomNumberGeneratorService.gen2 = process.RandomNumberGeneratorService.generator.clone()
process.gen1 = cms.EDFilter("Pythia8GeneratorFilter",
                            comEnergy = cms.double(7000.),
                            PythiaParameters = cms.PSet(
                                pythia8_example02 = cms.vstring('HardQCD:all = on',
                                                                'PhaseSpace:pTHatMin = 20.'),
                                parameterSets = cms.vstring('pythia8_example02')
                            )
                        )

_pythia8 = cms.EDFilter("Pythia8GeneratorFilter",
    comEnergy = cms.double(7000.),
    PythiaParameters = cms.PSet(
        pythia8_example02 = cms.vstring('HardQCD:all = on',
                                        'PhaseSpace:pTHatMin = 20.'),
        parameterSets = cms.vstring('pythia8_example02')
    )
)
from GeneratorInterface.Core.ExternalGeneratorFilter import ExternalGeneratorFilter
process.gen2 = ExternalGeneratorFilter(_pythia8)

process.sleeper = cms.EDProducer("timestudy::SleepingProducer", ivalue = cms.int32(1), consumes = cms.VInputTag(), eventTimes=cms.vdouble(1.))

process.compare = cms.EDAnalyzer("CompareGeneratorResultsAnalyzer", 
                                 module1 = cms.untracked.string("gen1"), 
                                 module2 =cms.untracked.string("gen2"),
                                 allowXSecDifferences = cms.untracked.bool(True))

process.p = cms.Path(process.sleeper+process.gen1+process.gen2+process.compare)
