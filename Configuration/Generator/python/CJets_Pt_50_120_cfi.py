import FWCore.ParameterSet.Config as cms

from Configuration.Generator.PythiaUESettings_cfi import *
generator = cms.EDFilter("Pythia6GeneratorFilter",
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    maxEventsToPrint = cms.untracked.int32(0),
    pythiaPylistVerbosity = cms.untracked.int32(0),
    filterEfficiency = cms.untracked.double(1.0),
    comEnergy = cms.double(10000.0),
    PythiaParameters = cms.PSet(
        pythiaUESettingsBlock,
        processParameters = cms.vstring('MSEL=0          ! User defined processes', 
            'MSUB(81)=1      ! qq->QQ massive', 
            'MSUB(82)=1      ! gg->QQ massive', 
            'MSTP(7)=4       ! 4 for CC_bar', 
            'CKIN(3)=50.     ! Pt hat lower cut', 
            'CKIN(4)=120.    ! Pt hat upper cut', 
            'CKIN(13)=0.     ! etamin', 
            'CKIN(14)=2.5    ! etamax', 
            'CKIN(15)=-2.5   ! -etamax', 
            'CKIN(16)=0.     ! -etamin'),
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring('pythiaUESettings', 
            'processParameters')
    )
)
