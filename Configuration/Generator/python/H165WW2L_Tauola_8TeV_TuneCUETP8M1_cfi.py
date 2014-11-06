import FWCore.ParameterSet.Config as cms
from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.Pythia8CUEP8M1Settings_cfi import *
from GeneratorInterface.ExternalDecays.TauolaSettings_cff import *
generator = cms.EDFilter("Pythia8GeneratorFilter",
                         pythiaPylistVerbosity = cms.untracked.int32(1),
                         # put here the efficiency of your filter (1. if no filter)
                         filterEfficiency = cms.untracked.double(1.0),
                         pythiaHepMCVerbosity = cms.untracked.bool(False),
                         # put here the cross section of your process (in pb)
                         crossSection = cms.untracked.double(0.325),
                         maxEventsToPrint = cms.untracked.int32(3),
                         comEnergy = cms.double(8000.0),
                         ExternalDecays = cms.PSet(
        Tauola = cms.untracked.PSet(
            TauolaPolar,
            TauolaDefaultInputCards
            ),
        parameterSets = cms.vstring('Tauola')
        ),
                         PythiaParameters = cms.PSet(
        pythia8CommonSettingsBlock,
        pythia8CUEP8M1SettingsBlock,
        processParameters = cms.vstring(
            'HiggsSM:gg2H = on',
            'HiggsSM:ff2Hff(t:WW) = on',
            'HiggsSM:ff2Hff(t:ZZ) = on ',
            '25:m0 = 165'
            '25:onMode = off',
            '25:onIfAny = 24 24',
            '24:onMode = off',
            '24:onMode = 11',
            '24:onMode = 13',
            '24:onMode = 15',
            ),
        parameterSets = cms.vstring('pythia8CommonSettings',
                                    'pythia8CUEP8M1Settings',
                                    'processParameters',
                                    )
        )
                         )

