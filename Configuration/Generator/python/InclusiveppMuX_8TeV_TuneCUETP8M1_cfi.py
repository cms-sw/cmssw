import FWCore.ParameterSet.Config as cms
from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.Pythia8CUEP8M1Settings_cfi import *

generator = cms.EDFilter("Pythia8GeneratorFilter",
                         pythiaHepMCVerbosity = cms.untracked.bool(False),
                         maxEventsToPrint = cms.untracked.int32(0),
                         pythiaPylistVerbosity = cms.untracked.int32(1),
                         filterEfficiency = cms.untracked.double(0.002305),
                         comEnergy = cms.double(8000.0),
                         crossSection = cms.untracked.double(51560000000.),
                         reweightGen = cms.PSet(),
                         PythiaParameters = cms.PSet(
        pythia8CommonSettingsBlock,
        pythia8CUEP8M1SettingsBlock,
        processParameters = cms.vstring(
            'HardQCD:all = on',
            'ParticleDecays:limitCylinder   = true',
            'ParticleDecays:xyMax = 1500',
            'ParticleDecays:zMax = 3000',
            '321:onMode = on', #kaons
            '211:onMode = on', #pion
            '130:onMode = on' #KL0
            ),
        parameterSets = cms.vstring('pythia8CommonSettings',
                                    'pythia8CUEP8M1Settings',
                                    'processParameters',
                                    )
        )
                         )

mugenfilter = cms.EDFilter("MCSmartSingleParticleFilter",
                           MinPt = cms.untracked.vdouble(2.5,2.5),
                           MinEta = cms.untracked.vdouble(-2.5,-2.5),
                           MaxEta = cms.untracked.vdouble(2.5,2.5),
                           ParticleID = cms.untracked.vint32(13,-13),
                           Status = cms.untracked.vint32(1,1),
                           # Decay cuts are in mm
                           MaxDecayRadius = cms.untracked.vdouble(1500.,1500.),
                           MinDecayZ = cms.untracked.vdouble(-3000.,-3000.),
                           MaxDecayZ = cms.untracked.vdouble(3000.,3000.)
)

ProductionFilterSequence = cms.Sequence(generator*mugenfilter)
