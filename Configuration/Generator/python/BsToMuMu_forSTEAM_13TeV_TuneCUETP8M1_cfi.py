import FWCore.ParameterSet.Config as cms
from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.Pythia8CUEP8M1Settings_cfi import *
from GeneratorInterface.EvtGenInterface.EvtGenSetting_cff import *

generator = cms.EDFilter("Pythia8GeneratorFilter",
                         comEnergy = cms.double(13000.0),
                         crossSection = cms.untracked.double(54000000000),
                         filterEfficiency = cms.untracked.double(3.0e-4),
                         pythiaHepMCVerbosity = cms.untracked.bool(False),
                         maxEventsToPrint = cms.untracked.int32(0),
                         pythiaPylistVerbosity = cms.untracked.int32(0),
                         ExternalDecays = cms.PSet(
        EvtGen130 = cms.untracked.PSet(
            decay_table = cms.string('GeneratorInterface/EvtGenInterface/data/DECAY_2010.DEC'),
            particle_property_file = cms.FileInPath('GeneratorInterface/EvtGenInterface/data/evt.pdl'),
          #  user_decay_file = cms.untracked.bool(True),
            user_decay_file = cms.vstring('GeneratorInterface/ExternalDecays/data/Bs_mumu.dec'),
            list_forced_decays = cms.vstring('MyB_s0','Myanti-B_s0'),
            operates_on_particles = cms.vint32()
            ),
        parameterSets = cms.vstring('EvtGen130')
        ),
                         #EvtGen = cms.untracked.PSet(
                         #operates_on_particles = cms.vint32( 0 ), # 0 (zero) means default list (hardcoded)
                         #use_default_decay = cms.untracked.bool(False),
                         #decay_table = cms.FileInPath('GeneratorInterface/ExternalDecays/data/DECAY_NOLONGLIFE.DEC'),
                         #particle_property_file = cms.FileInPath('GeneratorInterface/ExternalDecays/data/evt.pdl'),
                         #user_decay_file = cms.FileInPath('GeneratorInterface/ExternalDecays/data/Bs_mumu.dec'),
                         #list_forced_decays = cms.vstring('MyB_s0','Myanti-B_s0'),
                         #),
                         #parameterSets = cms.vstring('EvtGen')
                         #),
                         PythiaParameters = cms.PSet(
        pythia8CommonSettingsBlock,
        pythia8CUEP8M1SettingsBlock,
        processParameters = cms.vstring('HardQCD:all = on'),
        parameterSets = cms.vstring('pythia8CommonSettings',
                                    'pythia8CUEP8M1Settings',
                                    'processParameters',
                                    )
        )
                         )

generator.PythiaParameters.processParameters.extend(EvtGenExtraParticles)

MuMuFilter = cms.EDFilter("MCParticlePairFilter",
                          Status = cms.untracked.vint32(1, 1),
                          MinPt = cms.untracked.vdouble(3., 3.),
                          MaxEta = cms.untracked.vdouble(2.5, 2.5),
                          MinEta = cms.untracked.vdouble(-2.5, -2.5),
                          ParticleCharge = cms.untracked.int32(-1),
                          #MaxInvMass = cms.untracked.double(5.5),
                          #MinInvMass = cms.untracked.double(5.0),
                          ParticleID1 = cms.untracked.vint32(13),
                          ParticleID2 = cms.untracked.vint32(13)
                          )

# -- Require Muon from Bs
MuFilter = cms.EDFilter("PythiaFilter",
                        Status = cms.untracked.int32(1),
                        MotherID = cms.untracked.int32(531),
                        MinPt = cms.untracked.double(3.),
                        ParticleID = cms.untracked.int32(13),
                        MaxEta = cms.untracked.double(2.5),
                        MinEta = cms.untracked.double(-2.5)
                        )

ProductionFilterSequence = cms.Sequence(generator*MuMuFilter*MuFilter)
