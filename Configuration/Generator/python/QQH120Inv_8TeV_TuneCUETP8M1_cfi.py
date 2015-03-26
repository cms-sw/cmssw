import FWCore.ParameterSet.Config as cms
from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.Pythia8CUEP8M1Settings_cfi import *
generator = cms.EDFilter("Pythia8GeneratorFilter",
                         pythiaPylistVerbosity = cms.untracked.int32(1),
                         # put here the efficiency of your filter (1. if no filter)
                         filterEfficiency = cms.untracked.double(1.0),
                         pythiaHepMCVerbosity = cms.untracked.bool(False),
                         # put here the cross section of your process (in pb)
                         crossSection = cms.untracked.double(4.3),
                         comEnergy = cms.double(8000.0),
                         maxEventsToPrint = cms.untracked.int32(3),
                         PythiaParameters = cms.PSet(
        pythia8CommonSettingsBlock,
        pythia8CUEP8M1SettingsBlock,
        processParameters = cms.vstring( 'HiggsSM:ff2Hff(t:WW) = on',
                                         'HiggsSM:ff2Hff(t:ZZ) = on ',
                                         'HiggsSM:NLOWidths = on ',
                                         '25:m0 = 120',
                                         '25:onMode = off',
                                         '25:onIfAny = 23',
                                         '23:onMode = off',
                                         '23:onIfAny = 12'
                                         ),
        parameterSets = cms.vstring('pythia8CommonSettings',
                                    'pythia8CUEP8M1Settings',
                                    'processParameters',
                                    )
        )
                         )

