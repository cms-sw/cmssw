import FWCore.ParameterSet.Config as cms
from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.Pythia8CUEP8M1Settings_cfi import *
generator = cms.EDFilter("Pythia8GeneratorFilter",
                         pythiaPylistVerbosity = cms.untracked.int32(0),
                         filterEfficiency = cms.untracked.double(0.00013),
                         pythiaHepMCVerbosity = cms.untracked.bool(False),
                         crossSection = cms.untracked.double(54710000000.0),
                         maxEventsToPrint = cms.untracked.int32(0),
                         comEnergy = cms.double(8000.0),
                         PythiaParameters = cms.PSet(
        pythia8CommonSettingsBlock,
        pythia8CUEP8M1SettingsBlock,
        processParameters = cms.vstring(
            ' Bottomonium:all = on', # Quarkonia, MSEL=61
            'ParticleDecays:allowPhotonRadiation = on', # Turn on QED FSR
            ' ParticleDecays:mixB = off',
            '443:onMode = off', # Turn off J/psi decays
            '443:onIfMatch = 13 -13' # just let J/psi -> mu+ mu-
            ),
        parameterSets = cms.vstring('pythia8CommonSettings',
                                    'pythia8CUEP8M1Settings',
                                    'processParameters',
                                    )
        )
                         )

bfilter = cms.EDFilter("PythiaFilter",
                       ParticleID = cms.untracked.int32(5)
                       )

jpsifilter = cms.EDFilter("PythiaFilter",
                          Status = cms.untracked.int32(2),
                          MaxEta = cms.untracked.double(20.0),
                          MinEta = cms.untracked.double(-20.0),
                          MinPt = cms.untracked.double(0.0),
                          ParticleID = cms.untracked.int32(443)
                          )

mumugenfilter = cms.EDFilter("MCParticlePairFilter",
                             Status = cms.untracked.vint32(1, 1),
                             MinPt = cms.untracked.vdouble(0.5, 0.5),
                             MinP = cms.untracked.vdouble(2.7, 2.7),
                             MaxEta = cms.untracked.vdouble(2.5, 2.5),
                             MinEta = cms.untracked.vdouble(-2.5, -2.5),
                             ParticleCharge = cms.untracked.int32(-1),
                             MaxInvMass = cms.untracked.double(4.0),
                             MinInvMass = cms.untracked.double(2.0),
                             ParticleID1 = cms.untracked.vint32(13),
                             ParticleID2 = cms.untracked.vint32(13)
                             )

ProductionFilterSequence = cms.Sequence(generator*bfilter*jpsifilter*mumugenfilter)

