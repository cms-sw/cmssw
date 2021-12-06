import FWCore.ParameterSet.Config as cms
from Configuration.Generator.MCTunes2017.PythiaCP5Settings_cfi import *
from Configuration.Generator.Pythia8CommonSettings_cfi import *

generator = cms.EDFilter("Pythia8GeneratorFilter",
    pythiaPylistVerbosity = cms.untracked.int32(0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    maxEventsToPrint = cms.untracked.int32(0),
    comEnergy = cms.double(14000.0),
    ExternalDecays = cms.PSet(
        EvtGen130 = cms.untracked.PSet(
            decay_table = cms.string('GeneratorInterface/EvtGenInterface/data/DECAY_2014_NOLONGLIFE.DEC'),
            particle_property_file = cms.FileInPath('GeneratorInterface/EvtGenInterface/data/evt_2014.pdl'),
            operates_on_particles = cms.vint32(20443,445),                       # we care just about our signal particles
            convertPythiaCodes = cms.untracked.bool(False),
            user_decay_file = cms.vstring('GeneratorInterface/ExternalDecays/data/Onia_chic_jpsigamma.dec'),
            list_forced_decays = cms.vstring('Mychi_c1','Mychi_c2'),
        ),
        parameterSets = cms.vstring('EvtGen130')
    ),
    PythiaParameters = cms.PSet(
        pythia8CommonSettingsBlock,
        pythia8CP5SettingsBlock,
        processParameters = cms.vstring(
    # generate just the needed and nothing else
            'Charmonium:states(3PJ) = 20443,445',
            'Charmonium:O(3PJ)[3P0(1)] = 0.05,0.05',
            'Charmonium:O(3PJ)[3S1(8)] = 0.0031,0.0031',
            'Charmonium:gg2ccbar(3PJ)[3PJ(1)]g = on,on',
            'Charmonium:qg2ccbar(3PJ)[3PJ(1)]q = on,on',
            'Charmonium:qqbar2ccbar(3PJ)[3PJ(1)]g = on,on',
            'Charmonium:gg2ccbar(3PJ)[3S1(8)]g = on,on',
            'Charmonium:qg2ccbar(3PJ)[3S1(8)]q = on,on',
            'Charmonium:qqbar2ccbar(3PJ)[3S1(8)]g = on,on',
#
            'PhaseSpace:pTHatMin = 10.'                   # (filter efficiency 4.671e-02) be aware of this ckin(3) equivalent
            ),
        parameterSets = cms.vstring('pythia8CommonSettings',
                                    'pythia8CP5Settings',
                                    'processParameters',
                                    )
    )
)

# Next two muon filter are derived from muon reconstruction
oniafilter = cms.EDFilter("MCParticlePairFilter",
    Status = cms.untracked.vint32(2, 1),
    MinPt = cms.untracked.vdouble(16., 0.2),
    MaxEta = cms.untracked.vdouble(1.6, 1.6),
    MinEta = cms.untracked.vdouble(-1.6, -1.6),
    ParticleCharge = cms.untracked.int32(0),
    MinP = cms.untracked.vdouble(0.,0.),
    ParticleID1 = cms.untracked.vint32(443),
    ParticleID2 = cms.untracked.vint32(22)
)

muminusfilter = cms.EDFilter("PythiaDauVFilter",
    MotherID = cms.untracked.int32(0),
    MinPt = cms.untracked.vdouble(2.5, 2.5, 3.5),
    ParticleID = cms.untracked.int32(443),
    ChargeConjugation = cms.untracked.bool(False),
    MinEta = cms.untracked.vdouble(1.2, -1.6, -1.2),
    MaxEta = cms.untracked.vdouble(1.6, -1.2, 1.2),
    NumberDaughters = cms.untracked.int32(1),
    DaughterIDs = cms.untracked.vint32(-13, -13, -13)
)

muplusfilter = cms.EDFilter("PythiaDauVFilter",
    MotherID = cms.untracked.int32(0),
    MinPt = cms.untracked.vdouble(2.5, 2.5, 3.5),
    ParticleID = cms.untracked.int32(443),
    ChargeConjugation = cms.untracked.bool(False),
    MinEta = cms.untracked.vdouble(1.2, -1.6, -1.2),
    MaxEta = cms.untracked.vdouble(1.6, -1.2, 1.2),
    NumberDaughters = cms.untracked.int32(1),
    DaughterIDs = cms.untracked.vint32(13, 13, 13)
)

ProductionFilterSequence = cms.Sequence(generator*oniafilter*muminusfilter*muplusfilter)
