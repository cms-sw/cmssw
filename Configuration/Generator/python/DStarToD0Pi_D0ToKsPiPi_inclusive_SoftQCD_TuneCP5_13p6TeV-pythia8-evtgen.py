# Found 20 output events for 1500 input events.
# Filter efficiency = 0.013333
# Timing = 0.179750 sec/event
# Event size = 636.9 kB/event


import FWCore.ParameterSet.Config as cms
from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.MCTunesRun3ECM13p6TeV.PythiaCP5Settings_cfi import *
from GeneratorInterface.EvtGenInterface.EvtGenSetting_cff import *

_generator = cms.EDFilter("Pythia8GeneratorFilter",
    pythiaPylistVerbosity = cms.untracked.int32(0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    maxEventsToPrint = cms.untracked.int32(0),
    comEnergy = cms.double(13600.0),
    ExternalDecays = cms.PSet(
        EvtGen130 = cms.untracked.PSet(
            decay_table = cms.string('GeneratorInterface/EvtGenInterface/data/DECAY_2014_NOLONGLIFE.DEC'),
            particle_property_file = cms.FileInPath('GeneratorInterface/EvtGenInterface/data/evt_2014.pdl'),
            list_forced_decays = cms.vstring('MyDStar+','MyDStar-'),        
            operates_on_particles = cms.vint32(),    
            convertPythiaCodes = cms.untracked.bool(False),
            user_decay_embedded= cms.vstring(
"""
#
# Particles updated from PDG2018 https://journals.aps.org/prd/abstract/10.1103/PhysRevD.98.030001
Particle   pi+         1.3957061e-01   0.0000000e+00
Particle   pi-         1.3957061e-01   0.0000000e+00
Particle   K_S0        4.9761100e-01   0.0000000e+00
Particle   B-          5.2793200e+00   0.0000000e+00
Particle   B+          5.2793200e+00   0.0000000e+00
Particle   B0          5.2796300e+00   0.0000000e+00
Particle   anti-B0     5.2796300e+00   0.0000000e+00
Particle   B_s0        5.3668900e+00   0.0000000e+00
Particle   anti-B_s0   5.3668900e+00   0.0000000e+00
Particle   phi         1.0194610e+00   4.2490000e-03
Particle   D+          1.8696500e+00   0.0000000e+00
Particle   D-          1.8696500e+00   0.0000000e+00
Particle   D0          1.8648300e+00   0.0000000e+00
Particle   D*+         2.0102600e+00   0.0000834e-05
Particle   D*-         2.0102600e+00   0.0000834e-05
Particle   K+          4.9367700e-01   0.0000000e+00
Particle   K-          4.9367700e-01   0.0000000e+00

#
Alias      MyDStar+    D*+
Alias      MyDStar-    D*-
ChargeConj MyDStar-    MyDStar+
#
Alias      MyD0        D0
Alias      MyAntiD0    anti-D0
ChargeConj MyAntiD0    MyD0
#
Decay MyDStar+
1.000  MyD0  pi+    VSS;
Enddecay
CDecay MyDStar-
#
Decay MyD0
1.000  K_S0  pi+  pi-    PHSP;
Enddecay
CDecay MyAntiD0
#
End
"""
            )
        ),
        parameterSets = cms.vstring('EvtGen130')
    ),
    PythiaParameters = cms.PSet(
        pythia8CommonSettingsBlock,
        pythia8CP5SettingsBlock,
        processParameters = cms.vstring('SoftQCD:nonDiffractive = on',
					                    'PTFilter:filter = on', # this turn on the filter
                                        'PTFilter:quarkToFilter = 4', # PDG id of q quark
                                        'PTFilter:scaleToFilter = 2.0'
            ),
        parameterSets = cms.vstring('pythia8CommonSettings',
                                    'pythia8CP5Settings',
                                    'processParameters',
                                    )
    )
)

_generator.PythiaParameters.processParameters.extend(EvtGenExtraParticles)

from GeneratorInterface.Core.ExternalGeneratorFilter import ExternalGeneratorFilter
generator = ExternalGeneratorFilter(_generator)

###### Filters ##########
decayfilter = cms.EDFilter(
    "PythiaDauVFilter",
    verbose         = cms.untracked.int32(1),
    NumberDaughters = cms.untracked.int32(2),
    ParticleID      = cms.untracked.int32(413),  # DStar+
    DaughterIDs     = cms.untracked.vint32(421, 211), # D0 and pi+
    MinPt           = cms.untracked.vdouble(3.5, 0.1),
    MinEta          = cms.untracked.vdouble(-3., -3.),
    MaxEta          = cms.untracked.vdouble( 3.,  3.)
)

D0filter = cms.EDFilter(
    "PythiaDauVFilter",
    verbose         = cms.untracked.int32(1),
    NumberDaughters = cms.untracked.int32(2),
    MotherID        = cms.untracked.int32(413), # DStar+
    ParticleID      = cms.untracked.int32(421),  # D0
    DaughterIDs     = cms.untracked.vint32(310, 211, -211), # K0s pi+ pi-
    MinPt           = cms.untracked.vdouble(0.5, 0.1, 0.1),
    MinEta          = cms.untracked.vdouble(-3., -3., -3.),
    MaxEta          = cms.untracked.vdouble( 3.,  3.,  3.)
)
    

ProductionFilterSequence = cms.Sequence(generator*decayfilter*D0filter)
