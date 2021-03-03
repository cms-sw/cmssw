import FWCore.ParameterSet.Config as cms
from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.MCTunes2017.PythiaCP5Settings_cfi import *
from GeneratorInterface.EvtGenInterface.EvtGenSetting_cff import *

generator = cms.EDFilter("Pythia8GeneratorFilter",
                         pythiaPylistVerbosity = cms.untracked.int32(0),
                         pythiaHepMCVerbosity = cms.untracked.bool(False),
                         comEnergy = cms.double(13000.0),
                         maxEventsToPrint = cms.untracked.int32(0),
                         ExternalDecays = cms.PSet(
                         EvtGen130 = cms.untracked.PSet(
                            #uses latest evt and decay tables from evtgen
                            decay_table = cms.string('GeneratorInterface/EvtGenInterface/data/DECAY_NOLONGLIFE.DEC'),
                            particle_property_file = cms.FileInPath('GeneratorInterface/EvtGenInterface/data/evt.pdl'),
                            convertPythiaCodes = cms.untracked.bool(False),
                            #user_decay_file = cms.vstring('GeneratorInterface/ExternalDecays/data/Bu_Kstarmumu_Kspi.dec'),
                            #content was dump in the embed string below. This should test this feature.
                            list_forced_decays = cms.vstring('MyB+','MyB-'),
                            operates_on_particles = cms.vint32(),
                            user_decay_embedded= cms.vstring(
"""
# This is the decay file for the decay B+ -> MU+ MU- K*+(-> Ks pi+)
#
# Descriptor: [B+ -> mu+ mu- {,gamma} {,gamma} (K*+ -> Ks pi+)]cc
#
# NickName:
#
# Physics: Includes radiative mode
#
# Tested: Yes
# By:     K. Ulmer
# Date:   2-26-08
#
Alias      MyB+        B+
Alias      MyB-        B-
ChargeConj MyB+        MyB-
Alias      MyK*+       K*+
Alias      MyK*-       K*-
ChargeConj MyK*+       MyK*-
Alias      MyK_S0      K_S0
ChargeConj MyK_S0      MyK_S0
#
Decay MyB+
  1.000        MyK*+     mu+     mu-      BTOSLLBALL;
Enddecay
CDecay MyB-
#
Decay MyK*+
  1.000        MyK_S0    pi+              VSS;
Enddecay
CDecay MyK*-
#
Decay MyK_S0
  1.000        pi+       pi-              PHSP;
Enddecay
End
"""
                          ),
                ),
                parameterSets = cms.vstring('EvtGen130')
        ),
        PythiaParameters = cms.PSet(
        pythia8CommonSettingsBlock,
        pythia8CP5SettingsBlock,
        processParameters = cms.vstring(
            'HardQCD:gg2bbbar = on ',
            'HardQCD:qqbar2bbbar = on ',
            'HardQCD:hardbbbar = on',
            'PhaseSpace:pTHatMin = 20.',
            ),
        parameterSets = cms.vstring('pythia8CommonSettings',
                                    'pythia8CP5Settings',
                                    'processParameters',
                                    )
        )
                         )

generator.PythiaParameters.processParameters.extend(EvtGenExtraParticles)

configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    name = cms.untracked.string('$Source: Configuration/Generator/python/BuToKstarMuMu_forSTEAM_13TeV_cfi.py $'),
    annotation = cms.untracked.string('Summer14: Pythia8+EvtGen130 generation of Bu --> K* Mu+Mu-, 13TeV, Tune CP5')
    )

###########
# Filters #
###########
# Filter only pp events which produce a B+:
bufilter = cms.EDFilter("PythiaFilter", ParticleID = cms.untracked.int32(521))

# Filter on final state muons
mumugenfilter = cms.EDFilter("MCParticlePairFilter",
                             Status = cms.untracked.vint32(1, 1),
                             MinPt = cms.untracked.vdouble(2.8, 2.8),
                             MinP = cms.untracked.vdouble(2.8, 2.8),
                             MaxEta = cms.untracked.vdouble(2.3, 2.3),
                             MinEta = cms.untracked.vdouble(-2.3, -2.3),
                             ParticleID1 = cms.untracked.vint32(13,-13),
                             ParticleID2 = cms.untracked.vint32(13,-13)
                             )


ProductionFilterSequence = cms.Sequence(generator*bufilter*mumugenfilter)
