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
                'Define betas   0.015',
                'Define Apara   0.475',
                'Define Azero   0.724',
                'Define Aperp   0.500',
                'Define pApara  3.26',
                'Define pAzero  0.0',
                'Define pAperp  3.08',
                '#',
                'Alias      MyBs    B_s0',
                'Alias      Myanti-Bs   anti-B_s0',
                'ChargeConj Myanti-Bs   MyBs',
                '#',
                'Alias      MyJ/psi  J/psi',
                'Alias      MyPhi    phi',
                'ChargeConj MyJ/psi  MyJ/psi',
                'ChargeConj MyPhi    MyPhi',
                '#',
                'Decay MyBs',
                  '1.000         MyJ/psi     MyPhi        PVV_CPLH betas 1 Apara pApara Azero pAzero Aperp pAperp;',
                '#',
                'Enddecay',
                'Decay Myanti-Bs',
                  '1.000         MyJ/psi     MyPhi        PVV_CPLH betas 1 Apara pApara Azero pAzero Aperp pAperp;',
                'Enddecay',
                '#',
                'Decay MyJ/psi',
                  '1.000         mu+         mu-          PHOTOS VLL;',
                'Enddecay',
                '#',
                'Decay MyPhi',
                  '1.000         K+          K-           VSS;',
                'Enddecay',
                'End'
            ), 
            list_forced_decays = cms.vstring('MyBs','Myanti-Bs'),
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
    name = cms.untracked.string('$Source: /Configuration/Generator/pythonBsJpsiPhi_mumuKK_PhaseII_cfipy $'),
    annotation = cms.untracked.string('PhaseII: Pythia8+EvtGen130 generation of Bs --> Jpsi phi (mu mu KK), 14TeV, Tune CP5')
)

bfilter = cms.EDFilter(
    "PythiaFilter",
    MaxEta = cms.untracked.double(9999.),
    MinEta = cms.untracked.double(-9999.),
    ParticleID = cms.untracked.int32(531)
)

jpsifilter = cms.EDFilter(
    "PythiaDauVFilter",
    MotherID = cms.untracked.int32(531),
    ParticleID = cms.untracked.int32(443),
    NumberDaughters = cms.untracked.int32(2),
    DaughterIDs = cms.untracked.vint32(13, -13),
    MinPt = cms.untracked.vdouble(2.5, 2.5),
    MinEta = cms.untracked.vdouble(-2.9, -2.9),
    MaxEta = cms.untracked.vdouble(2.9, 2.9),
    verbose = cms.untracked.int32(1)
)

phifilter = cms.EDFilter(
    "PythiaDauVFilter",
    MotherID = cms.untracked.int32(531),
    ParticleID = cms.untracked.int32(333),
    NumberDaughters = cms.untracked.int32(2),
    DaughterIDs = cms.untracked.vint32(321, -321),
    MinPt = cms.untracked.vdouble(0.4, 0.4),
    MinEta = cms.untracked.vdouble(-4.1, -4.1),
    MaxEta = cms.untracked.vdouble(4.1, 4.1),
    verbose = cms.untracked.int32(1)
)

ProductionFilterSequence = cms.Sequence(generator*bfilter*jpsifilter*phifilter)
