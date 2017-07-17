import FWCore.ParameterSet.Config as cms

from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.Pythia8CUEP8M1Settings_cfi import *


generator = cms.EDFilter("Pythia8GeneratorFilter",
        comEnergy = cms.double(14000.0),
        crossSection = cms.untracked.double(2.0221e+09),
        filterEfficiency = cms.untracked.double(1.0),
        maxEventsToPrint = cms.untracked.int32(1),
        pythiaHepMCVerbosity = cms.untracked.bool(False),
        pythiaPylistVerbosity = cms.untracked.int32(1),
        #reweightGen = cms.bool(True), #

        PythiaParameters = cms.PSet(
                pythia8CommonSettingsBlock,
                pythia8CUEP8M1SettingsBlock,
                processParameters = cms.vstring(
                        'HardQCD:all = on',
                        'PhaseSpace:pTHatMin = 15',
                        'PhaseSpace:pTHatMax = 7000',
                        'PhaseSpace:bias2Selection = on',
                        'PhaseSpace:bias2SelectionPow = 6.0',
                        'PhaseSpace:bias2SelectionRef = 15.',

                ),
                parameterSets = cms.vstring('pythia8CommonSettings',
                                            'pythia8CUEP8M1Settings',
                                            'processParameters'
                                            )
        )
)
