import FWCore.ParameterSet.Config as cms
from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.Pythia8CUEP8M1Settings_cfi import *
generator = cms.EDFilter("Pythia8GeneratorFilter",
                         pythiaHepMCVerbosity = cms.untracked.bool(False),
                         maxEventsToPrint = cms.untracked.int32(0),
                         pythiaPylistVerbosity = cms.untracked.int32(0),
                         filterEfficiency = cms.untracked.double(1.0),
                         comEnergy = cms.double(8000.0),
                         PythiaParameters = cms.PSet(
        pythia8CommonSettingsBlock,
        pythia8CUEP8M1SettingsBlock,
        processParameters = cms.vstring(
            'HiggsSM:gg2H = on',
            'HiggsSM:ff2Hff(t:WW) = on',
            'HiggsSM:ff2Hff(t:ZZ) = on ',
            '25:m0 = 200'
            '25:onMode = off',
            '25:onIfAny = 23 23',
            '23:onMode = off',
            '23:onMode = 11',
            '23:onMode = 13',
            '23:onMode = 15',
            'PhaseSpace:mHatMinSecond = 5 ',
            'PhaseSpace:mHatMaxSecond = 150'
            ),
        parameterSets = cms.vstring('pythia8CommonSettings',
                                    'pythia8CUEP8M1Settings',
                                    'processParameters',
                                    )
        )
                         )

