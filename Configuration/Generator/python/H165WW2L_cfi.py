# The following comments couldn't be translated into the new config version:

# "MSTJ(41)=1              !Switch off Pythia QED bremsshtrahlung",

# W decays

# Higgs decays

import FWCore.ParameterSet.Config as cms

from Configuration.Generator.PythiaUESettings_cfi import *
generator = cms.EDFilter("Pythia6GeneratorFilter",
    pythiaPylistVerbosity = cms.untracked.int32(1),
    # put here the efficiency of your filter (1. if no filter)
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    # put here the cross section of your process (in pb)
    crossSection = cms.untracked.double(0.325),
    maxEventsToPrint = cms.untracked.int32(3),
    comEnergy = cms.double(10000.0),
    PythiaParameters = cms.PSet(
        pythiaUESettingsBlock,
        processParameters = cms.vstring('PMAS(25,1)=165.0        !mass of Higgs', 
            'MSEL=0                  ! user selection for process', 
            'MSUB(102)=1             !ggH', 
            'MSUB(123)=1             !ZZ fusion to H', 
            'MSUB(124)=1             !WW fusion to H', 
            'MDME(190,1) = 0            !W decay into dbar u', 
            'MDME(191,1) = 0            !W decay into dbar c', 
            'MDME(192,1) = 0            !W decay into dbar t', 
            'MDME(194,1) = 0            !W decay into sbar u', 
            'MDME(195,1) = 0            !W decay into sbar c', 
            'MDME(196,1) = 0            !W decay into sbar t', 
            'MDME(198,1) = 0            !W decay into bbar u', 
            'MDME(199,1) = 0            !W decay into bbar c', 
            'MDME(200,1) = 0            !W decay into bbar t', 
            'MDME(206,1) = 1            !W decay into e+ nu_e', 
            'MDME(207,1) = 1            !W decay into mu+ nu_mu', 
            'MDME(208,1) = 1            !W decay into tau+ nu_tau', 
            'MDME(210,1)=0           !Higgs decay into dd', 
            'MDME(211,1)=0           !Higgs decay into uu', 
            'MDME(212,1)=0           !Higgs decay into ss', 
            'MDME(213,1)=0           !Higgs decay into cc', 
            'MDME(214,1)=0           !Higgs decay into bb', 
            'MDME(215,1)=0           !Higgs decay into tt', 
            'MDME(216,1)=0           !Higgs decay into', 
            'MDME(217,1)=0           !Higgs decay into Higgs decay', 
            'MDME(218,1)=0           !Higgs decay into e nu e', 
            'MDME(219,1)=0           !Higgs decay into mu nu mu', 
            'MDME(220,1)=0           !Higgs decay into tau nu tau', 
            'MDME(221,1)=0           !Higgs decay into Higgs decay', 
            'MDME(222,1)=0           !Higgs decay into g g', 
            'MDME(223,1)=0           !Higgs decay into gam gam', 
            'MDME(224,1)=0           !Higgs decay into gam Z', 
            'MDME(225,1)=0           !Higgs decay into Z Z', 
            'MDME(226,1)=1           !Higgs decay into W W'),
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring('pythiaUESettings', 
            'processParameters')
    )
)
