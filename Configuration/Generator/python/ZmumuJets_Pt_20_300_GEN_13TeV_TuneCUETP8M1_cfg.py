import FWCore.ParameterSet.Config as cms
from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.Pythia8CUEP8M1Settings_cfi import *

generator = cms.EDFilter("Pythia8GeneratorFilter",
                         #pythiaHepMCVerbosity = cms.untracked.bool(False),
                         maxEventsToPrint = cms.untracked.int32(0),
                         pythiaPylistVerbosity = cms.untracked.int32(0),
                         filterEfficiency = cms.untracked.double(1.0),
                         # cross section: 20-30 == 326.7, 30-50 == 227.0, 50-80 == 93.17,
                         # 80-120 == 31.48, 120-170 == 9.63, 170-230 == 2.92, 230-300 == 0.8852
                         # crossSection = cms.untracked.double(691.7852),
                         #
                         # at 10 TeV it scales down to 426
                         #
                         crossSection = cms.untracked.double(425.6),
                         comEnergy = cms.double(13000.0),
                         PythiaParameters = cms.PSet(
        pythia8CommonSettingsBlock,
        pythia8CUEP8M1SettingsBlock,
        processParameters = cms.vstring(
            'WeakSingleBoson:ffbar2gmZ = on',
            '23:onMode = off',
            '23:onIfAny = 13',
            'PhaseSpace:pTHatMin = 20.',
            'PhaseSpace:pTHatMax = 300.',
            'PhaseSpace:mHatMin = 75.',
            ),
        parameterSets = cms.vstring('pythia8CommonSettings',
                                    'pythia8CUEP8M1Settings',
                                    'processParameters',
                                    )
        )
                         )
ProductionFilterSequence = cms.Sequence(generator)
