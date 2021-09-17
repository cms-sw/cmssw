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
                'Alias      MyB0        B0',
                'Alias      Myanti-B0   anti-B0',
                'ChargeConj MyB0        Myanti-B0',
                'Alias      MyK*0       K*0',
                'Alias      MyK*0bar    anti-K*0',
                'ChargeConj MyK*0       MyK*0bar',
                '#',
                'Decay MyB0',
                '1.000        MyK*0     mu+     mu-               BTOSLLBALL;',
                'Enddecay',
                'Decay Myanti-B0',
                '1.000        MyK*0bar     mu+     mu-            BTOSLLBALL;',
                'Enddecay',
                '#',
                'Decay MyK*0',
                '1.000        K+        pi-                    VSS;',
                'Enddecay',
                'Decay MyK*0bar',
                '1.000        K-        pi+                    VSS;',
                'Enddecay ',
                'End'
            ), 
            list_forced_decays = cms.vstring('MyB0','Myanti-B0'),
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
    name = cms.untracked.string('$Source: /Configuration/Generator/python/BPH_BsMuMu_PhaseII_cfi.py $'),
    annotation = cms.untracked.string('PhaseII: Pythia8+EvtGen130 generation of B0 --> K* mu+mu-, 14TeV, Tune CP5')
)

bfilter = cms.EDFilter(
    "PythiaFilter", 
    MaxEta = cms.untracked.double(9999.),
    MinEta = cms.untracked.double(-9999.),
    ParticleID = cms.untracked.int32(511)  ## Bd
    )

decayfilter = cms.EDFilter(
    "PythiaDauVFilter",
    verbose         = cms.untracked.int32(1),
    NumberDaughters = cms.untracked.int32(3),
    ParticleID      = cms.untracked.int32(511),
    DaughterIDs     = cms.untracked.vint32(-13, 13, 313),  ## mu+, mu-, K*^0(892)
    MinPt           = cms.untracked.vdouble(2.5, 2.5, -1.),
    MinEta          = cms.untracked.vdouble(-2.9, -2.9, -9999.),
    MaxEta          = cms.untracked.vdouble( 2.9,  2.9,  9999.)
    )

kstarfilter = cms.EDFilter(
    "PythiaDauVFilter",
    verbose         = cms.untracked.int32(1), 
    NumberDaughters = cms.untracked.int32(2), 
    MotherID        = cms.untracked.int32(511),  ## Bd
    ParticleID      = cms.untracked.int32(313),  ## K*^0(892)
    DaughterIDs     = cms.untracked.vint32(321, -211), ## K+, pi-
    MinPt           = cms.untracked.vdouble(0.4, 0.4), 
    MinEta          = cms.untracked.vdouble(-4.1, -4.1), 
    MaxEta          = cms.untracked.vdouble( 4.1,  4.1)
    )

ProductionFilterSequence = cms.Sequence(generator*bfilter*decayfilter*kstarfilter)
