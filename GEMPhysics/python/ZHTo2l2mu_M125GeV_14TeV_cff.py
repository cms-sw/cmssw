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
            'MSUB(24)= 1       ! ff->ZH (SM)', 
            'PMAS(25,1)= 125.  ! m_h',
            'PMAS(6,1)= 173.3  ! mass of top quark',
            'PMAS(23,1)=91.187 ! mass of Z',
            'PMAS(24,1)=80.39  ! mass of W',
            # Z decay                                    
            'MDME(182,1)=1     ! Z decay into e- e+',
            'MDME(184,1)=1     ! Z decay into mu- mu+',
            'MDME(186,1)=1     ! Z decay into tau- tau+',
            # Higgs boson decays
            'MDME(218,1)=0     ! Higgs decay into e- e+', 
            'MDME(219,1)=1     ! Higgs decay into mu- mu+', 
            'MDME(220,1)=0     ! Higgs decay into tau- tau+', 
        ),                            
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring(
            'pythiaUESettings', 
            'processParameters')
    )
)

