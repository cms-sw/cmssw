import FWCore.ParameterSet.Config as cms
from Configuration.Generator.PythiaUEZ2starSettings_cfi import *
from GeneratorInterface.ExternalDecays.TauolaSettings_cff import *

generator = cms.EDFilter("Pythia6GeneratorFilter",
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    comEnergy = cms.double(13000.0),
    maxEventsToPrint = cms.untracked.int32(0),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(1.0),
    crossSection = cms.untracked.double(0.00002497),
    ExternalDecays = cms.PSet(
        Tauola = cms.untracked.PSet(
            TauolaPolar,
            TauolaDefaultInputCards
        ),
        parameterSets = cms.vstring('Tauola')
    ),
    PythiaParameters = cms.PSet(
        # Default (mostly empty - to keep PYTHIA default) card file
        # Name of the set is "pythiaDefault"
        pythiaUESettingsBlock,
        # User cards - name is "myParameters"
        processParameters = cms.vstring('MSEL=0             ! User defined processes', 
            "MSUB(141)   = 1    ! ff -> gamma/Z0/Z\'", 
            'MSTP(44) = 3       ! only select the Z process', 
            "PMAS(32,1)  = 2250 ! Z\' mass (GeV)", 
            'CKIN(1)     = -1  ! lower invariant mass cutoff (GeV)', 
            'CKIN(2)     = -1   ! no upper invariant mass cutoff', 
            'PARU(121)=  0.        ! vd', 
            'PARU(122)=  0.506809  ! ad', 
            'PARU(123)=  0.        ! vu', 
            'PARU(124)=  0.506809  ! au', 
            'PARU(125)=  0.        ! ve', 
            'PARU(126)=  0.506809  ! ae', 
            'PARU(127)= -0.253405  ! vnu', 
            'PARU(128)=  0.253405  ! anu', 
            'PARJ(180)=  0.        ! vd', 
            'PARJ(181)=  0.506809  ! ad', 
            'PARJ(182)=  0.        ! vu', 
            'PARJ(183)=  0.506809  ! au', 
            'PARJ(184)=  0.        ! ve', 
            'PARJ(185)=  0.506809  ! ae', 
            'PARJ(186)= -0.253405  ! vnu', 
            'PARJ(187)=  0.253405  ! anu', 
            'PARJ(188)=  0.        ! vd', 
            'PARJ(189)=  0.506809  ! ad', 
            'PARJ(190)=  0.        ! vu', 
            'PARJ(191)=  0.506809  ! au', 
            'PARJ(192)=  0.        ! ve', 
            'PARJ(193)=  0.506809  ! ae', 
            'PARJ(194)= -0.253405  ! vnu', 
            'PARJ(195)=  0.253405  ! anu', 
            'MDME(289,1) = 0    ! d dbar', 
            'MDME(290,1) = 0    ! u ubar', 
            'MDME(291,1) = 0    ! s sbar', 
            'MDME(292,1) = 0    ! c cbar', 
            'MDME(293,1) = 0    ! b bar', 
            'MDME(294,1) = 0    ! t tbar', 
            'MDME(295,1) = -1   ! 4th gen q qbar', 
            'MDME(296,1) = -1   ! 4th gen q qbar', 
            'MDME(297,1) = 1    ! e-     e+', 
            'MDME(298,1) = 0    ! nu_e   nu_ebar', 
            'MDME(299,1) = 0    ! mu-    mu+', 
            'MDME(300,1) = 0    ! nu_mu  nu_mubar', 
            'MDME(301,1) = 0    ! tau    tau', 
            'MDME(302,1) = 0    ! nu_tau nu_taubar', 
            'MDME(303,1) = -1   ! 4th gen l- l+', 
            'MDME(304,1) = -1   ! 4th gen nu nubar', 
            'MDME(305,1) = -1   ! W+ W-', 
            'MDME(306,1) = -1   ! H+ H-', 
            'MDME(307,1) = -1   ! Z0 gamma', 
            'MDME(308,1) = -1   ! Z0 h0', 
            'MDME(309,1) = -1   ! h0 A0', 
            'MDME(310,1) = -1   ! H0 A0'),
        parameterSets = cms.vstring('pythiaUESettings','processParameters')
    )
)


ProductionFilterSequence = cms.Sequence(generator)        
        
