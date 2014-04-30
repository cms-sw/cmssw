import FWCore.ParameterSet.Config as cms

from Configuration.Generator.PythiaUEZ2Settings_cfi import *
from GeneratorInterface.ExternalDecays.TauolaSettings_cff import *

generator = cms.EDFilter("Pythia6GeneratorFilter",
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    maxEventsToPrint = cms.untracked.int32(0),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(1.0),
    crossSection = cms.untracked.double(1.504*0.0632),
    comEnergy = cms.double(14000.0),
    ExternalDecays = cms.PSet(
        Tauola = cms.untracked.PSet(
            TauolaPolar,
            TauolaDefaultInputCards
        ),
        parameterSets = cms.vstring('Tauola')
    ),                    
    UseExternalGenerators = cms.untracked.bool(True),
    PythiaParameters = cms.PSet(
    pythiaUESettingsBlock,             
# set proccess to be simulated
    processParameters = cms.vstring(
            'MSEL=0            ! User defined processes', 
            'MSUB(26)= 1       ! gg->WH (SM)', 
            'PMAS(25,1)= 125.  ! m_h',
            'PMAS(6,1)= 172.6  ! mass of top quark',
            'PMAS(23,1)=91.187 ! mass of Z',
            'PMAS(24,1)=80.39  ! mass of W',
# W decay                                    
            'MDME(190,1)=0   !W decay into dbar u', 
            'MDME(191,1)=0   !W decay into dbar c', 
            'MDME(192,1)=0   !W decay into dbar t', 
            'MDME(194,1)=0   !W decay into sbar u', 
            'MDME(195,1)=0   !W decay into sbar c', 
            'MDME(196,1)=0   !W decay into sbar t', 
            'MDME(198,1)=0   !W decay into bbar u', 
            'MDME(199,1)=0   !W decay into bbar c', 
            'MDME(200,1)=0   !W decay into bbar t', 
            'MDME(205,1)=0   !W decay into bbar tp', 
            'MDME(206,1)=1   !W decay into e+ nu_e', 
            'MDME(207,1)=1   !W decay into mu+ nu_mu', 
            'MDME(208,1)=1   !W decay into tau+ nu_tau',                                                                      
# Higgs boson decays
            'MDME(210,1)=0   !Higgs decay into dd', 
            'MDME(211,1)=0   !Higgs decay into uu', 
            'MDME(212,1)=0   !Higgs decay into ss', 
            'MDME(213,1)=0   !Higgs decay into cc', 
            'MDME(214,1)=0   !Higgs decay into bb', 
            'MDME(215,1)=0   !Higgs decay into tt', 
            'MDME(216,1)=0   !Higgs decay into', 
            'MDME(217,1)=0   !Higgs decay into Higgs decay', 
            'MDME(218,1)=0   !Higgs decay into e nu e', 
            'MDME(219,1)=0   !Higgs decay into mu nu mu', 
            'MDME(220,1)=0   !Higgs decay into tau tau', 
            'MDME(221,1)=0   !Higgs decay into Higgs decay', 
            'MDME(222,1)=1   !Higgs decay into g g', 
            'MDME(223,1)=0   !Higgs decay into gam gam', 
            'MDME(224,1)=0   !Higgs decay into gam Z', 
            'MDME(225,1)=0   !Higgs decay into Z Z', 
            'MDME(226,1)=0   !Higgs decay into W W'
        ),                            
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring('pythiaUESettings', 
            'processParameters')
    )
)

