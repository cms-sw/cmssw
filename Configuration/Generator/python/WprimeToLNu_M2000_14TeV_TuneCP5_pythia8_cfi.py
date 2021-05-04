import FWCore.ParameterSet.Config as cms
from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.MCTunes2017.PythiaCP5Settings_cfi import *

generator = cms.EDFilter("Pythia8GeneratorFilter",
                         pythiaHepMCVerbosity = cms.untracked.bool(False),
                         maxEventsToPrint = cms.untracked.int32(0),
                         pythiaPylistVerbosity = cms.untracked.int32(1),
                         filterEfficiency = cms.untracked.double(1.),
                         crossSection = cms.untracked.double(1.),
                         comEnergy = cms.double(14000.0),
                         PythiaParameters = cms.PSet(
        pythia8CommonSettingsBlock,
        pythia8CP5SettingsBlock,
        processParameters = cms.vstring(
            'NewGaugeBoson:ffbar2Wprime = on',
            '34:m0 = 2000 ',
            '34:onMode = off',
            '34:onIfAny = 11 12',
            '34:onIfAny = 13 14',
            '34:onIfAny = 15 16',
            ),
        parameterSets = cms.vstring('pythia8CommonSettings',
                                    'pythia8CP5Settings',
                                    'processParameters',
                                )
                         )
                     )

ProductionFilterSequence = cms.Sequence(generator)
