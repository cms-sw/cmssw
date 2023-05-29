# Found 43 output events for 15000 input events.
# Filter efficiency = 0.002867
# Timing = 0.377745 sec/event
# Event size = 501.9 kB/event



import FWCore.ParameterSet.Config as cms
from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.MCTunesRun3ECM13p6TeV.PythiaCP5Settings_cfi import *
from GeneratorInterface.EvtGenInterface.EvtGenSetting_cff import *

_generator = cms.EDFilter("Pythia8GeneratorFilter",
                         comEnergy = cms.double(13600.0),
                         crossSection = cms.untracked.double(54000000000),
                         filterEfficiency = cms.untracked.double(3.0e-4),
                         pythiaHepMCVerbosity = cms.untracked.bool(False),
                         maxEventsToPrint = cms.untracked.int32(0),
                         pythiaPylistVerbosity = cms.untracked.int32(0),
#                          maxEventsToPrint = cms.untracked.int32(1),
#                          pythiaPylistVerbosity = cms.untracked.int32(12),
                         ExternalDecays = cms.PSet(
        EvtGen130 = cms.untracked.PSet(
##            decay_table = cms.string('GeneratorInterface/EvtGenInterface/data/DECAY_2010_NOLONGLIFE.DEC'),
            decay_table = cms.string('GeneratorInterface/EvtGenInterface/data/DECAY_2014_NOLONGLIFE.DEC'),
#            particle_property_file = cms.FileInPath('GeneratorInterface/EvtGenInterface/data/evt.pdl'),
            particle_property_file = cms.FileInPath('GeneratorInterface/EvtGenInterface/data/evt_2014.pdl'),
            #user_decay_file = cms.vstring('GeneratorInterface/ExternalDecays/data/Bs_Jpsiphi.dec'),
            user_decay_embedded= cms.vstring(
'#',
'# Particles updated from PDG2018 https://journals.aps.org/prd/abstract/10.1103/PhysRevD.98.030001',
'Particle   pi+           1.3957061e-01   0.0000000e+00',
'Particle   pi-           1.3957061e-01   0.0000000e+00',
'Particle   mu+           1.0565837e-01   0.0000000e+00', ## id 13
'Particle   mu-           1.0565837e-01   0.0000000e+00',
'Particle   K+            4.9367700e-01   0.0000000e+00', ## id 321
'Particle   K-            4.9367700e-01   0.0000000e+00',
'Particle   p+            9.3827203e-01   0.0000000e+00', ## id 2212
'Particle   anti-p-       9.3827203e-01   0.0000000e+00',
'Particle   K_S0          4.9761100e-01   0.0000000e+00', ## id 310
'Particle   K*+           8.9176000e-01   5.0300000e-02',
'Particle   K*-           8.9176000e-01   5.0300000e-02',
'Particle   K*0           8.9555000e-01   4.7300000e-02',
'Particle   anti-K*0      8.9555000e-01   4.7300000e-02',
'Particle   rho0          7.7526000e-01   1.4910000e-01',
'Particle   phi           1.0194610e+00   4.2490000e-03',
'Particle   Lambda0       1.1156830e+00   0.0000000e+00', ## id 3122
'Particle   anti-Lambda0  1.1156830e+00   0.0000000e+00',
'Particle   Sigma0        1.1926420e+00   8.8947595e-06', ## id 3212
'Particle   anti-Sigma0   1.1926420e+00   8.8947595e-06',
'Particle   B-            5.2793200e+00   0.0000000e+00',
'Particle   B+            5.2793200e+00   0.0000000e+00',
'Particle   B0            5.2796300e+00   0.0000000e+00',
'Particle   anti-B0       5.2796300e+00   0.0000000e+00',
'Particle   B_s0          5.3668900e+00   0.0000000e+00',
'Particle   anti-B_s0     5.3668900e+00   0.0000000e+00',
'Particle   J/psi         3.0969000e+00   9.2900006e-05', ## id 443
'Particle   psi(2S)       3.6860970e+00   2.9400000e-04', ## id 100443
'Particle   Lambda_b0     5.6196000e+00   0.0000000e-04', ## id 5122,
'Particle anti-Lambda_b0  5.6196000e+00   0.0000000e-04',
'#',
'#',
'#',
'Alias      MyLb        Lambda_b0',
'Alias      Myanti-Lb   anti-Lambda_b0',
'ChargeConj Myanti-Lb   MyLb',
'#',
'Alias       Mypsi      J/psi',
'ChargeConj  Mypsi      Mypsi',
'#',
'Alias      MyLam        Lambda0',
'Alias      Myanti-Lam   anti-Lambda0',
'ChargeConj Myanti-Lam  MyLam',
'#',
'Decay Mypsi',
'1.000       mu+    mu-        PHOTOS VLL;',
'Enddecay',
'#',
'Decay MyLb',
'1.000       Mypsi  MyLam      PHSP;',
'Enddecay',
'CDecay Myanti-Lb',
'End'
), 
            list_forced_decays = cms.vstring('MyLb','Myanti-Lb'),
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
            "5122:m0=5.6196",       ## changing also lambda_b0 mass in pythia
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

#
lbfilter = cms.EDFilter("PythiaFilter", ParticleID = cms.untracked.int32(5122))

# psifilter = cms.EDFilter("PythiaFilter",
#         MotherID        = cms.untracked.int32(5122),
#         ParticleID      = cms.untracked.int32(443)
# #         MinPt           = cms.untracked.double(4.95),
# #         MinEta          = cms.untracked.double(-3.0),
# #         MaxEta          = cms.untracked.double( 3.0)
# )
psifilter = cms.EDFilter("PythiaDauVFilter",
        verbose         = cms.untracked.int32(0),
        NumberDaughters = cms.untracked.int32(2),
        MotherID        = cms.untracked.int32(5122),
        ParticleID      = cms.untracked.int32(443),
        DaughterIDs     = cms.untracked.vint32(13, -13),
        MinPt           = cms.untracked.vdouble(3., 3.),
        MinEta          = cms.untracked.vdouble(-2.5, -2.5),
        MaxEta          = cms.untracked.vdouble(2.5, 2.5)
)
decayfilter = cms.EDFilter("PythiaDauVFilter",
	    verbose         = cms.untracked.int32(0),
	    NumberDaughters = cms.untracked.int32(2),
	    MotherID        = cms.untracked.int32(0),
	    ParticleID      = cms.untracked.int32(5122),
        DaughterIDs     = cms.untracked.vint32(443, 3122),
	    MinPt           = cms.untracked.vdouble(5, 0.5),
	    MinEta          = cms.untracked.vdouble(-99999, -3),
	    MaxEta          = cms.untracked.vdouble( 99999,  3)
)


# ProductionFilterSequence = cms.Sequence(generator*lbfilter*psifilter)
ProductionFilterSequence = cms.Sequence(generator*lbfilter*decayfilter*psifilter)
