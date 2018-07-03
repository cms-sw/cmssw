import FWCore.ParameterSet.Config as cms

from Configuration.Generator.Pythia8CommonSettings_cfi import * 
from Configuration.Generator.Pythia8CUEP8M1Settings_cfi import * 

generator = cms.EDFilter("Pythia8GeneratorFilter",
                         comEnergy = cms.double(13000.0),
                         crossSection = cms.untracked.double(0.013),
                         filterEfficiency = cms.untracked.double(1),
                         maxEventsToPrint = cms.untracked.int32(1),
                         pythiaHepMCVerbosity = cms.untracked.bool(False),
                         pythiaPylistVerbosity = cms.untracked.int32(1),
                         PythiaParameters = cms.PSet(
        pythia8CommonSettingsBlock,
        pythia8CUEP8M1SettingsBlock,
        processParameters = cms.vstring(
            'NewGaugeBoson:ffbar2Wprime = on',
            '34:m0 = 3000',
            '34:onMode = off',
            '34:onIfAny = 11 13 15',
            ),
        parameterSets = cms.vstring(
            'pythia8CommonSettings',
            'pythia8CUEP8M1Settings',
            'processParameters')
        )
                         )
ProductionFilterSequence = cms.Sequence(generator)
