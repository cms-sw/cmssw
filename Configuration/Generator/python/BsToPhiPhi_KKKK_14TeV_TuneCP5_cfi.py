import FWCore.ParameterSet.Config as cms

from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.MCTunes2017.PythiaCP5Settings_cfi import *
from GeneratorInterface.EvtGenInterface.EvtGenSetting_cff import *

generator = cms.EDFilter("Pythia8GeneratorFilter",
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    maxEventsToPrint = cms.untracked.int32(0),
    pythiaPylistVerbosity = cms.untracked.int32(0),
    filterEfficiency = cms.untracked.double(1.38e-3),
    crossSection = cms.untracked.double(540000000.),
    comEnergy = cms.double(14000.0),
    ExternalDecays = cms.PSet(
        EvtGen130 = cms.untracked.PSet(
            decay_table = cms.string('GeneratorInterface/EvtGenInterface/data/DECAY_2010.DEC'),
            particle_property_file = cms.FileInPath('GeneratorInterface/EvtGenInterface/data/evt.pdl'),
            user_decay_embedded= cms.vstring(
                'Define Hp 0.49',
                'Define Hz 0.775',
                'Define Hm 0.4',
                'Define pHp 2.50',
                'Define pHz 0.0',
                'Define pHm -0.17',
                '#',
                'Alias      MyB_s0   B_s0',
                'Alias      Myanti-B_s0   anti-B_s0',
                'ChargeConj Myanti-B_s0   MyB_s0',
                'Alias      MyPhi    phi',
                'ChargeConj MyPhi    MyPhi',
                '#',
                'Decay MyB_s0',
                '  1.000         MyPhi      MyPhi        PVV_CPLH 0.02 1 Hp pHp Hz pHz Hm pHm;',
                '#',
                'Enddecay',
                'Decay Myanti-B_s0',
                 ' 1.000         MyPhi      MyPhi        PVV_CPLH 0.02 1 Hp pHp Hz pHz Hm pHm;',
                'Enddecay',
                '#',
                'Decay MyPhi',
                 ' 1.000         K+          K-           VSS;',
                'Enddecay',
                'End'
            ), 
            list_forced_decays = cms.vstring('MyB_s0','Myanti-B_s0'),
            operates_on_particles = cms.vint32(),
        ),
        parameterSets = cms.vstring('EvtGen130')
    ),
    PythiaParameters = cms.PSet(pythia8CommonSettingsBlock,
        pythia8CP5SettingsBlock,                        
        processParameters = cms.vstring(
            "SoftQCD:nonDiffractive = on",
            'PTFilter:filter = on', # this turn on the filter
            'PTFilter:quarkToFilter = 5', # PDG id of q quark
            'PTFilter:scaleToFilter = 1.0'
        ),
        parameterSets = cms.vstring('pythia8CommonSettings',
            'pythia8CP5Settings',                        
            'processParameters',
        )
    )
)

generator.PythiaParameters.processParameters.extend(EvtGenExtraParticles)

configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.0 $'),
    name = cms.untracked.string('$Source: /Configuration/Generator/python/BsPhiPhi_KKKK_PhaseII_cfi.py $'),
    annotation = cms.untracked.string('PhaseII: Pythia8+EvtGen130 generation of Bs --> phi phi (4K), 14TeV, Tune CP5')
)

bfilter = cms.EDFilter(
    "PythiaFilter",
    MaxEta = cms.untracked.double(9999.),
    MinEta = cms.untracked.double(-9999.),
    ParticleID = cms.untracked.int32(531)
)

decayfilter = cms.EDFilter(
    "PythiaDauVFilter",
    verbose         = cms.untracked.int32(1),
    NumberDaughters = cms.untracked.int32(2),
    ParticleID      = cms.untracked.int32(531),
    DaughterIDs     = cms.untracked.vint32(333, 333),  ## phi phi
    MinPt           = cms.untracked.vdouble(3.0, 3.0),
    MinEta          = cms.untracked.vdouble(-3.0, -3.0),
    MaxEta          = cms.untracked.vdouble( 3.0,  3.0)
)

phifilter = cms.EDFilter(
    "PythiaDauVFilter",
    verbose = cms.untracked.int32(0),
    NumberDaughters = cms.untracked.int32(2),
    MotherID = cms.untracked.int32(531),
    ParticleID = cms.untracked.int32(333),
    DaughterIDs = cms.untracked.vint32(321, -321),
    MinPt = cms.untracked.vdouble(1.95, 1.95),
    MinEta = cms.untracked.vdouble(-2.5, -2.5),
    MaxEta = cms.untracked.vdouble(2.5, 2.5)
)

ProductionFilterSequence = cms.Sequence(generator*bfilter*decayfilter*phifilter)
