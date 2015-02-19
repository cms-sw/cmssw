import FWCore.ParameterSet.Config as cms
from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.Pythia8CUEP8M1Settings_cfi import *

generator = cms.EDFilter("Pythia8GeneratorFilter",
                         pythiaHepMCVerbosity = cms.untracked.bool(False),
                         maxEventsToPrint = cms.untracked.int32(0),
                         pythiaPylistVerbosity = cms.untracked.int32(0),
                         filterEfficiency = cms.untracked.double(1.0),
                         comEnergy = cms.double(13000.0),
                         PythiaParameters = cms.PSet(
        pythia8CommonSettingsBlock,
        pythia8CUEP8M1SettingsBlock,
        processParameters = cms.vstring(
            'WeakBosonAndParton:qqbar2Wg = on',
            'WeakBosonAndParton:qqbar2Wg = on',
            '24:onMode = off',
            '24:onIfAny = 11 12',
            '24:onIfAny = 13 14',
            '24:onIfAny = 15 16',
            'PhaseSpace:pTHatMin = 3000.',
            'PhaseSpace:pTHatMax = 3500.',
            ),
        parameterSets = cms.vstring('pythia8CommonSettings',
                                    'pythia8CUEP8M1Settings',
                                    'processParameters',
                                    )
        )
                         )
ProductionFilterSequence = cms.Sequence(generator)

