# Only for Trigger Study

import FWCore.ParameterSet.Config as cms

from Configuration.Generator.PythiaUEZ2starSettings_cfi import *

generator = cms.EDFilter(
    "Pythia6GeneratorFilter",
    comEnergy = cms.double(13000.0),
    crossSection = cms.untracked.double(54000000000),
    filterEfficiency = cms.untracked.double(0.0013),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    maxEventsToPrint = cms.untracked.int32(0),
    pythiaPylistVerbosity = cms.untracked.int32(0),
    ExternalDecays = cms.PSet(
        EvtGen = cms.untracked.PSet(
             operates_on_particles = cms.vint32( 0 ), # 0 (zero) means default list (hardcoded)
                                                      # you can put here the list of particles (PDG IDs)
                                                      # that you want decayed by EvtGen
             use_default_decay = cms.untracked.bool(False),
             decay_table = cms.FileInPath('GeneratorInterface/ExternalDecays/data/DECAY_NOLONGLIFE.DEC'),
             particle_property_file = cms.FileInPath('GeneratorInterface/ExternalDecays/data/evt.pdl'),
             user_decay_file = cms.FileInPath('GeneratorInterface/ExternalDecays/data/Bd_mumu.dec'),
             list_forced_decays = cms.vstring('MyB0',
                                              'Myanti-B0'),
        ),
        parameterSets = cms.vstring('EvtGen')
    ),

    
    PythiaParameters = cms.PSet(
    pythiaUESettingsBlock,
         bbbarSettings = cms.vstring('MSEL = 1'),
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring(
             'pythiaUESettings',
             'bbbarSettings')
       
    )
)


MuMuFilter = cms.EDFilter("MCParticlePairFilter",
    Status = cms.untracked.vint32(1, 1),
    MinPt = cms.untracked.vdouble(3., 3.),
    MaxEta = cms.untracked.vdouble(2.5, 2.5),
    MinEta = cms.untracked.vdouble(-2.5, -2.5),
    ParticleCharge = cms.untracked.int32(-1),
    MaxInvMass = cms.untracked.double(5.5),
    MinInvMass = cms.untracked.double(5.0),
    ParticleID1 = cms.untracked.vint32(13),
    ParticleID2 = cms.untracked.vint32(13)
)

# -- Require Muon from Bd
MuFilter = cms.EDFilter("PythiaFilter",
    Status = cms.untracked.int32(1),
    MotherID = cms.untracked.int32(511),
    MinPt = cms.untracked.double(3.),
    ParticleID = cms.untracked.int32(13),
    MaxEta = cms.untracked.double(2.5),
    MinEta = cms.untracked.double(-2.5)
)

ProductionFilterSequence = cms.Sequence(generator*MuMuFilter*MuFilter)
