import FWCore.ParameterSet.Config as cms

from Configuration.Generator.PythiaUEZ2starSettings_cfi import *

generator = cms.EDFilter("Pythia6GeneratorFilter",
    pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    comEnergy = cms.double(8000.0),
    crossSection = cms.untracked.double(17.52),
    maxEventsToPrint = cms.untracked.int32(0),
    PythiaParameters = cms.PSet(
        pythiaUESettingsBlock,
        processParameters = cms.vstring('PMAS(6,1)=172.3 ! t quark mass', 
            'PMAS(347,1)= 750.0 ! graviton mass', 
            'PARP(50)=0.54  ! c(k/Mpl) * 5.4', 
            'MSEL=0         ! User defined processes', 
            'MSUB(391)=1    ! ffbar->G*', 
            'MSUB(392)=1    ! gg->G*', 
            '5000039:ALLOFF ! Turn off graviton decays', 
            '5000039:ONIFANY 1 2 3 4 5 11 13 ! graviton decays into quarks (except top) and leptons ', 
            'CKIN(3)=25.    ! Pt hat lower cut', 
            'CKIN(4)=-1.    ! Pt hat upper cut', 
            'CKIN(13)=-10.  ! etamin', 
            'CKIN(14)=10.   ! etamax', 
            'CKIN(15)=-10.  ! -etamax', 
            'CKIN(16)=10.   ! -etamin'),
        parameterSets = cms.vstring('pythiaUESettings', 
            'processParameters')
    )
)
