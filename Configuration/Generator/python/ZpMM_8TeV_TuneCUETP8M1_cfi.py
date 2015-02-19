import FWCore.ParameterSet.Config as cms
from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.Pythia8CUEP8M1Settings_cfi import *

generator = cms.EDFilter("Pythia8GeneratorFilter",
                         pythiaHepMCVerbosity = cms.untracked.bool(False),
                         comEnergy = cms.double(8000.0),
                         maxEventsToPrint = cms.untracked.int32(0),
                         pythiaPylistVerbosity = cms.untracked.int32(1),
                         filterEfficiency = cms.untracked.double(1.0),
                         crossSection = cms.untracked.double(0.00002497),
                         PythiaParameters = cms.PSet(
        pythia8CommonSettingsBlock,
        pythia8CUEP8M1SettingsBlock,
        processParameters = cms.vstring(
            'NewGaugeBoson:ffbar2gmZZprime = on',
            'Zprime:gmZmode = 3',
            'Zprime:vd =      0.0',
            'Zprime:ad =      0.506809',
            'Zprime:vu =      0.0',
            'Zprime:au =      0.506809',
            'Zprime:ve =      0.0',
            'Zprime:ae =      0.506809',
            'Zprime:vnue =   -0.253405',
            'Zprime:anue =    0.253405',
            'Zprime:vs =      0.0',
            'Zprime:as =      0.506809',
            'Zprime:vc =      0.0',
            'Zprime:ac =      0.506809',
            'Zprime:vmu =     0.0',
            'Zprime:amu =     0.506809',
            'Zprime:vnumu =  -0.253405',
            'Zprime:anumu =   0.253405',
            'Zprime:vb =      0.0',
            'Zprime:ab =      0.506809',
            'Zprime:vt =      0.0',
            'Zprime:at =      0.506809',
            'Zprime:vtau =    0.0',
            'Zprime:atau =    0.506809',
            'Zprime:vnutau = -0.253405',
            'Zprime:anutau =  0.253405',
            '32:m0 =3000',
            '32:onMode = off',
            '32:onIfAny = 13',
            ),
        parameterSets = cms.vstring('pythia8CommonSettings',
                                    'pythia8CUEP8M1Settings',
                                    'processParameters',
                                    )
        )
                         )

ProductionFilterSequence = cms.Sequence(generator)

