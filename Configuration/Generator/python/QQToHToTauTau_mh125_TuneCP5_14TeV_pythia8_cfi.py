import FWCore.ParameterSet.Config as cms
from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.MCTunes2017.PythiaCP5Settings_cfi import *
generator = cms.EDFilter("Pythia8GeneratorFilter",
                         pythiaPylistVerbosity = cms.untracked.int32(1),
                         # put here the efficiency of your filter (1. if no filter)
                         filterEfficiency = cms.untracked.double(1.0),
                         comEnergy = cms.double(14000.0),
                         pythiaHepMCVerbosity = cms.untracked.bool(False),
                         # put here the cross section of your process (in pb)
                         crossSection = cms.untracked.double(0.388),
                         maxEventsToPrint = cms.untracked.int32(0),
                         PythiaParameters = cms.PSet(
        pythia8CommonSettingsBlock,
        pythia8CP5SettingsBlock,
        processParameters = cms.vstring(
            #'HiggsSM:gg2H = on',
            'HiggsSM:ff2Hff(t:WW) = on',
            'HiggsSM:ff2Hff(t:ZZ) = on',
            '25:m0 = 125',
            '25:onMode = off',
            '25:onIfAny = 15',
            ),
        parameterSets = cms.vstring('pythia8CommonSettings',
                                    'pythia8CP5Settings',
                                    'processParameters',
                                    )
        )
                         )

