import FWCore.ParameterSet.Config as cms
process = cms.Process("TEST")

process.source =cms.Source("EmptySource")

process.maxEvents.input = 10

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
process.gen2 = cms.EDFilter("Pythia8GeneratorFilter",
                            comEnergy = cms.double(7000.),
                            PythiaParameters = cms.PSet(
                                pythia8_example02 = cms.vstring('HardQCD:all = on',
                                                                'PhaseSpace:pTHatMin = 20.'),
                                parameterSets = cms.vstring('pythia8_example02')
                            )
                        )
process.compare = cms.EDAnalyzer("CompareGeneratorResultsAnalyzer", module1 = cms.untracked.string("gen1"), module2 =cms.untracked.string("gen2"))

process.p = cms.Path(process.gen1+process.gen2+process.compare)
