import FWCore.ParameterSet.Config as cms
from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.MCTunes2017.PythiaCP5Settings_cfi import *

generator = cms.EDFilter("Pythia8GeneratorFilter",
                         pythiaHepMCVerbosity = cms.untracked.bool(False),
                         comEnergy = cms.double(14000.0),
                         maxEventsToPrint = cms.untracked.int32(0),
                         pythiaPylistVerbosity = cms.untracked.int32(1),
                         filterEfficiency = cms.untracked.double(1.0),
                         crossSection = cms.untracked.double(0.004783), # 4.233 pb * 0.00113 (qqH * BR(4v))
                         PythiaParameters = cms.PSet(
        pythia8CommonSettingsBlock,
        pythia8CP5SettingsBlock,
        processParameters = cms.vstring(
            'HiggsSM:ff2Hff(t:ZZ) = on',
            'HiggsSM:ff2Hff(t:WW) = on',
            '25:m0 = 125',
            '25:onMode = off',
            '25:onIfMatch = 23 23',
            '23:onMode = off',
            '23:onIfAny = 12 14 16',
            ),
        parameterSets = cms.vstring('pythia8CommonSettings',
                                    'pythia8CP5Settings',
                                    'processParameters',
                                    )
        )
                         )
