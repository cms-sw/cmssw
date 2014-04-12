# The following comments couldn't be translated into the new config version:

# Higgs decays

import FWCore.ParameterSet.Config as cms

from Configuration.Generator.PythiaUESettings_cfi import *
generator = cms.EDFilter("Pythia6GeneratorFilter",
    pythiaPylistVerbosity = cms.untracked.int32(1),
    UseTauola = cms.untracked.bool(False),
    # put here the efficiency of your filter (1. if no filter)
    filterEfficiency = cms.untracked.double(1.0),
    comEnergy = cms.double(10000.0),
    UseTauolaPolarization = cms.untracked.bool(False),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
#    GeneratorParameters = cms.PSet(
#        parameterSets = cms.vstring('generator'),
#        generator = cms.vstring('TAUOLA = 0 0 0   ! TAUOLA ')
#    ),
    #untracked bool UseTauola = false
    #untracked bool UseTauolaPolarization = false
    # put here the cross section of your process (in pb)
    crossSection = cms.untracked.double(0.388),
    maxEventsToPrint = cms.untracked.int32(3),
    PythiaParameters = cms.PSet(
        pythiaUESettingsBlock,
        processParameters = cms.vstring('PMAS(25,1)=135.0        !mass of Higgs', 
            'MSEL=0                  !user selection for process', 
            'MSUB(123)=1             !ZZ fusion to H', 
            'MSUB(124)=1             !WW fusion to H', 
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
            'MDME(220,1)=1           !Higgs decay into tau tau', 
            'MDME(221,1)=0           !Higgs decay into Higgs decay', 
            'MDME(222,1)=0           !Higgs decay into g g', 
            'MDME(223,1)=0           !Higgs decay into gam gam', 
            'MDME(224,1)=0           !Higgs decay into gam Z', 
            'MDME(225,1)=0           !Higgs decay into Z Z', 
            'MDME(226,1)=0           !Higgs decay into W W'),
        parameterSets = cms.vstring('pythiaUESettings', 
            'processParameters')
    )
)
