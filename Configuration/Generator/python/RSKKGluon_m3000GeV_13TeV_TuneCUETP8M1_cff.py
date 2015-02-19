import FWCore.ParameterSet.Config as cms
from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.Pythia8CUEP8M1Settings_cfi import *


generator = cms.EDFilter("Pythia8GeneratorFilter",
                         crossSection = cms.untracked.double(1.0),
                         pythiaPylistVerbosity = cms.untracked.int32(0),
                         filterEfficiency = cms.untracked.double(1.0),
                         pythiaHepMCVerbosity = cms.untracked.bool(False),
                         comEnergy = cms.double(13000.0),
                         maxEventsToPrint = cms.untracked.int32(0),
                         PythiaParameters = cms.PSet(
        pythia8CommonSettingsBlock,
        pythia8CUEP8M1SettingsBlock,
        processParameters = cms.vstring(
            'ExtraDimensionsG*:qqbar2KKgluon* = on',
            'ExtraDimensionsG*:KKintMode = 2',
            'ExtraDimensionsG*:KKgqR = -0.2',
            'ExtraDimensionsG*:KKgqL = -0.2',
            'ExtraDimensionsG*:KKgbR = -0.2',
            'ExtraDimensionsG*:KKgbL = 1.0',
            'ExtraDimensionsG*:KKgtR = 5.0',
            'ExtraDimensionsG*:KKgtL = 1.0',
            '5100021:m0 = 3000'),
        parameterSets = cms.vstring('pythia8CommonSettings',
                                    'pythia8CUEP8M1Settings',
                                    'processParameters',
                                    )
        )
                         )

ProductionFilterSequence = cms.Sequence(generator)


