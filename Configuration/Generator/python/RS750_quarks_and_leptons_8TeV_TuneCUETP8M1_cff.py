import FWCore.ParameterSet.Config as cms
from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.Pythia8CUEP8M1Settings_cfi import *
generator = cms.EDFilter("Pythia8GeneratorFilter",
                         pythiaPylistVerbosity = cms.untracked.int32(1),
                         filterEfficiency = cms.untracked.double(1.0),
                         pythiaHepMCVerbosity = cms.untracked.bool(False),
                         comEnergy = cms.double(8000.0),
                         crossSection = cms.untracked.double(17.52),
                         maxEventsToPrint = cms.untracked.int32(0),
                         PythiaParameters = cms.PSet(
        pythia8CommonSettingsBlock,
        pythia8CUEP8M1SettingsBlock,
        processParameters = cms.vstring(
            'ExtraDimensionsG*:gg2G* = on',
            'ExtraDimensionsG*:ffbar2G* = on'
            '6:m0 = 172.3 ', 
            '39:m0 = 750.0 ',
            'ExtraDimensionsG*:kappaMG = 0.54 ',
            '39:onmode = off',
            '39:onIfAny = 1 2 3 4 5 11 13 ',
            'PhaseSpace:pTHatMin = 25',
            'PhaseSpace:pTHatMax = -1', 
            #'CKIN(13)=-10.  ! etamin', 
            #'CKIN(14)=10.   ! etamax', 
            #'CKIN(15)=-10.  ! -etamax', 
            #'CKIN(16)=10.   ! -etamin'
            ),
        parameterSets = cms.vstring('pythia8CommonSettings',
                                    'pythia8CUEP8M1Settings',
                                    'processParameters',
                                    )
        
        )
                         )

