import FWCore.ParameterSet.Config as cms
from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.MCTunes2017.PythiaCP5Settings_cfi import *
from GeneratorInterface.EvtGenInterface.EvtGenSetting_cff import *

generator = cms.EDFilter("Pythia8GeneratorFilter",
                         comEnergy = cms.double(14000.0),
                         pythiaHepMCVerbosity = cms.untracked.bool(False),
                         maxEventsToPrint = cms.untracked.int32(0),
                         pythiaPylistVerbosity = cms.untracked.int32(0),
                         ExternalDecays = cms.PSet(
        EvtGen130 = cms.untracked.PSet(
            decay_table = cms.string('GeneratorInterface/EvtGenInterface/data/DECAY_2014_NOLONGLIFE.DEC'),
            particle_property_file = cms.FileInPath('GeneratorInterface/EvtGenInterface/data/evt_2014.pdl'),
            user_decay_embedded= cms.vstring(
"""## my decay
Alias      MyB+    B+
Alias      MyB-   B-
ChargeConj MyB-   MyB+
#
Alias      MyJpsi J/psi
ChargeConj MyJpsi MyJpsi
#
# Alias      Myrho0       rho0
# ChargeConj Myrho0       Myrho0
#
#
Alias       Mypsi(2S)  psi(2S)
ChargeConj  Mypsi(2S)  Mypsi(2S)
#
Decay MyJpsi
1.000  mu+       mu-                     PHOTOS VLL;
Enddecay
#
Decay MyB+
1.000     Mypsi(2S) K+         PHSP;
Enddecay
#
Decay MyB-
1.000     Mypsi(2S) K-         PHSP;
Enddecay
#
Decay Mypsi(2S)
1.000   MyJpsi  pi+   pi-         VVPIPI;
Enddecay
#
# Decay Myrho0
# 1.000   pi+   pi-                  PHSP;
# Enddecay
#
End"""
     ),
            list_forced_decays = cms.vstring('MyB+','MyB-'),
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



###########
# Filters #
###########
bfilter = cms.EDFilter("PythiaFilter",
        MaxEta = cms.untracked.double(9999.),
        MinEta = cms.untracked.double(-9999.),
        ParticleID = cms.untracked.int32(521)
)

jpsifilter = cms.EDFilter("PythiaDauVFilter",
        verbose         = cms.untracked.int32(0), 
        NumberDaughters = cms.untracked.int32(2), 
        MotherID        = cms.untracked.int32(100443),  
        ParticleID      = cms.untracked.int32(443),  
        DaughterIDs     = cms.untracked.vint32(13, -13),
        MinPt           = cms.untracked.vdouble(1.5, 1.5), 
        MinEta          = cms.untracked.vdouble(-2.5, -2.5), 
        MaxEta          = cms.untracked.vdouble( 2.5,  2.5)
)

xxxfilter = cms.EDFilter("PythiaDauVFilter",
        verbose         = cms.untracked.int32(0), 
        NumberDaughters = cms.untracked.int32(3), 
        MotherID        = cms.untracked.int32(521),  
        ParticleID      = cms.untracked.int32(100443),  
        DaughterIDs     = cms.untracked.vint32(443, 211, -211),
        MinPt           = cms.untracked.vdouble(5, 0.0, 0.0), 
        MinEta          = cms.untracked.vdouble(-9999, -9999, -9999), 
        MaxEta          = cms.untracked.vdouble( 9999,  9999,  9999)
)

decayfilter = cms.EDFilter("PythiaDauVFilter",
        verbose         = cms.untracked.int32(0), 
        NumberDaughters = cms.untracked.int32(2), 
        MotherID        = cms.untracked.int32(0),  
        ParticleID      = cms.untracked.int32(521),  
        DaughterIDs     = cms.untracked.vint32(100443, 321),
        MinPt           = cms.untracked.vdouble(5., 0.0), 
        MinEta          = cms.untracked.vdouble(-9999., -9999.), 
        MaxEta          = cms.untracked.vdouble( 9999.,  9999.)
)

ProductionFilterSequence = cms.Sequence(generator*bfilter*jpsifilter*xxxfilter*decayfilter)
