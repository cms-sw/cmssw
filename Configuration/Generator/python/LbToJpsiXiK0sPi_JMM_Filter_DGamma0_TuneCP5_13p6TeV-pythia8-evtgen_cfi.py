# Found 48 output events for 8000 input events.
# Filter efficiency = 0.006
# Timing = 0.260728 sec/event
# Event size = 546.5 kB/event

import FWCore.ParameterSet.Config as cms
from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.MCTunesRun3ECM13p6TeV.PythiaCP5Settings_cfi import *
from GeneratorInterface.EvtGenInterface.EvtGenSetting_cff import *

_generator = cms.EDFilter("Pythia8GeneratorFilter",
                         comEnergy = cms.double(13600.0),
                         crossSection = cms.untracked.double(54000000000),
                         filterEfficiency = cms.untracked.double(3.0e-4),
                         #pythiaHepMCVerbosity = cms.untracked.bool(False),
                         maxEventsToPrint = cms.untracked.int32(0),
                         pythiaPylistVerbosity = cms.untracked.int32(0),
                         ExternalDecays = cms.PSet(
        EvtGen130 = cms.untracked.PSet(
            decay_table = cms.string('GeneratorInterface/EvtGenInterface/data/DECAY_2014_NOLONGLIFE.DEC'),
            particle_property_file = cms.FileInPath('GeneratorInterface/EvtGenInterface/data/evt_2014.pdl'),
            user_decay_embedded= cms.vstring(
'#',
'# Particles updated from PDG2020 published in Prog. Theor. Exp. Phys. 2020, 083C01 (2020)',
'Particle   pi+           1.3957039e-01   0.0000000e+00',
'Particle   pi-           1.3957039e-01   0.0000000e+00',
'Particle   Xi-           1.3217100e+00   0.0000000e+00', ## id 3312
'Particle   anti-Xi+      1.3217100e+00   0.0000000e+00', ## id -3312
'Particle   Lambda0       1.1156830e+00   0.0000000e+00', ## id 3122
'Particle   anti-Lambda0  1.1156830e+00   0.0000000e+00', ## id -3122
'Particle   Lambda_b0     5.6196000e+00   0.0000000e+00', ## id 5122
'Particle   anti-Lambda_b0 5.6196000e+00   0.0000000e+00', ## id -5122
'Particle   K+            4.9367700e-01   0.0000000e+00', ## id 321
'Particle   K-            4.9367700e-01   0.0000000e+00',
'Particle   K_S0          4.9761100e-01   0.0000000e+00', ## id 310
'Particle   J/psi         3.0969000e+00   9.2900000e-05',
'#',
'Alias      MyLambda_b0    Lambda_b0',
'Alias      Myanti-Lambda_b0   anti-Lambda_b0',
'ChargeConj Myanti-Lambda_b0   MyLambda_b0',
'Alias      MyJ/psi  J/psi',
'ChargeConj MyJ/psi  MyJ/psi',
'#',
'Alias       MyK0s      K_S0',
'ChargeConj  MyK0s      MyK0s',
'#',
'Decay MyLambda_b0',
'1.000      MyJ/psi     Xi-         MyK0s    pi+  PHSP;',
'Enddecay',
'Decay Myanti-Lambda_b0',
'1.000      MyJ/psi     anti-Xi+    MyK0s    pi-  PHSP;',
'Enddecay',
'#',
'Decay MyJ/psi',
'1.000         mu+         mu-          PHOTOS VLL;',
'Enddecay',
'#',
'End'),
            list_forced_decays = cms.vstring('MyLambda_b0','Myanti-Lambda_b0'),
            operates_on_particles = cms.vint32(),
            convertPythiaCodes = cms.untracked.bool(False)
            ),
        parameterSets = cms.vstring('EvtGen130')
        ),
        PythiaParameters = cms.PSet(
        pythia8CommonSettingsBlock,
        pythia8CP5SettingsBlock,
        processParameters = cms.vstring(
            "SoftQCD:nonDiffractive = on",
            "5122:m0=5.61960",     ## changing also Lambda_b0 mass in pythia
            'PTFilter:filter = on', # this turn on the filter
            'PTFilter:quarkToFilter = 5', # PDG id of q quark
            'PTFilter:scaleToFilter = 1.0'),
        parameterSets = cms.vstring(
            'pythia8CommonSettings',
            'pythia8CP5Settings',
            'processParameters',
        )
    )
)
_generator.PythiaParameters.processParameters.extend(EvtGenExtraParticles)

from GeneratorInterface.Core.ExternalGeneratorFilter import ExternalGeneratorFilter
generator = ExternalGeneratorFilter(_generator)

###########
# Filters #
###########
# Filter only pp events which produce a Lambdab:
lbfilter = cms.EDFilter("PythiaFilter", ParticleID = cms.untracked.int32(5122))

decayfilter = cms.EDFilter(
        "PythiaDauVFilter",
  verbose         = cms.untracked.int32(0),
  NumberDaughters = cms.untracked.int32(3),
  MotherID        = cms.untracked.int32(0),
  ParticleID      = cms.untracked.int32(5122),
  DaughterIDs     = cms.untracked.vint32(443, 3312, 310, 211),
  MinPt           = cms.untracked.vdouble(2.5, 0.7, 0.5, 0.1),
  MinEta          = cms.untracked.vdouble(-9999., -3.0, -3.0, -3.0),
  MaxEta          = cms.untracked.vdouble( 9999.,  3.0,  3.0,  3.0)
        )

jpsifilter = cms.EDFilter("PythiaDauVFilter",
  verbose         = cms.untracked.int32(0),
  NumberDaughters = cms.untracked.int32(2),
  MotherID        = cms.untracked.int32(5122),
  ParticleID      = cms.untracked.int32(443),
  DaughterIDs     = cms.untracked.vint32(13, -13),
  MinPt           = cms.untracked.vdouble(1., 1.),
  MinEta          = cms.untracked.vdouble(-3., -3.),
  MaxEta          = cms.untracked.vdouble( 3.,  3.)
          )

ProductionFilterSequence = cms.Sequence(generator*lbfilter*decayfilter*jpsifilter)
