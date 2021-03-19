import FWCore.ParameterSet.Config as cms
from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.MCTunes2017.PythiaCP5Settings_cfi import *
from GeneratorInterface.EvtGenInterface.EvtGenSetting_cff import *

generator = cms.EDFilter("Pythia8GeneratorFilter",
                         comEnergy = cms.double(13000.0),
                         crossSection = cms.untracked.double(54000000000),
                         filterEfficiency = cms.untracked.double(3.0e-4),
                         pythiaHepMCVerbosity = cms.untracked.bool(False),
                         maxEventsToPrint = cms.untracked.int32(0),
                         pythiaPylistVerbosity = cms.untracked.int32(0),
                         ExternalDecays = cms.PSet(
                         #using alternative name for decayer
                         EvtGen1 = cms.untracked.PSet(
	                    #uses latest evt and decay tables from evtgen 
                            decay_table = cms.string('GeneratorInterface/EvtGenInterface/data/DECAY_NOLONGLIFE.DEC'),
                            particle_property_file = cms.FileInPath('GeneratorInterface/EvtGenInterface/data/evt.pdl'),
                            convertPythiaCodes = cms.untracked.bool(False),
                            #here we will use the user.dec store in the release
                            user_decay_file = cms.vstring('GeneratorInterface/ExternalDecays/data/Bs_mumu.dec'),
                            list_forced_decays = cms.vstring('MyB_s0','Myanti-B_s0'),
                            operates_on_particles = cms.vint32()
                         ),
                         parameterSets = cms.vstring('EvtGen1')
                        ),
                        PythiaParameters = cms.PSet(
                           pythia8CommonSettingsBlock,
                           pythia8CP5SettingsBlock,
                           processParameters = cms.vstring(
                              #filter of a b-quark before hadronizing, and use a better data-like process
                              'PTFilter:filter = on',
                              'PTFilter:quarkToFilter = 5',
                              'PTFilter:scaleToFilter = 1.0',
                              'SoftQCD:nonDiffractive = on',
                           ),
                           parameterSets = cms.vstring('pythia8CommonSettings',
                                    'pythia8CP5Settings',
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
                          ParticleID1 = cms.untracked.vint32(13,-13),
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
