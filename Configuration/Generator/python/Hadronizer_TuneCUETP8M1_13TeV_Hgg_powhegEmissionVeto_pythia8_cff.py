import FWCore.ParameterSet.Config as cms
from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.Pythia8CUEP8M1Settings_cfi import *
from Configuration.Generator.Pythia8PowhegEmissionVetoSettings_cfi import *

generator = cms.EDFilter("Pythia8ConcurrentHadronizerFilter",
                         maxEventsToPrint = cms.untracked.int32(1),
                         pythiaPylistVerbosity = cms.untracked.int32(1),
                         filterEfficiency = cms.untracked.double(1.0),
                         pythiaHepMCVerbosity = cms.untracked.bool(False),
                         comEnergy = cms.double(13000.),
                         PythiaParameters = cms.PSet(
             pythia8CommonSettingsBlock,
             pythia8CUEP8M1SettingsBlock,
             pythia8PowhegEmissionVetoSettingsBlock,
             processParameters = cms.vstring(
                'POWHEG:nFinal = 1', ## Number of final state particles
                                     ## (BEFORE THE DECAYS) in the LHE
                                     ## other than emitted extra parton
                '25:m0 = 125.0',
                '25:onMode = off',
                '25:onIfMatch = 22 22',
              ),
             parameterSets = cms.vstring('pythia8CommonSettings',
                                         'pythia8CUEP8M1Settings',
                                         'pythia8PowhegEmissionVetoSettings',
                                         'processParameters'
                                         )
             )
                            )
ProductionFilterSequence = cms.Sequence(generator)
