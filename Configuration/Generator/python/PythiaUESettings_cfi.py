import FWCore.ParameterSet.Config as cms

pythiaUESettingsBlock = cms.PSet(
    pythiaUESettings = cms.vstring(
        'MSTJ(11)=3     ! Choice of the fragmentation function', 
        'MSTJ(22)=2     ! Decay those unstable particles', 
        'PARJ(71)=10 .  ! for which ctau  10 mm', 
        'MSTP(2)=1      ! which order running alphaS', 
        'MSTP(33)=0     ! no K factors in hard cross sections', 
        'MSTP(51)=10042 ! structure function chosen (external PDF CTEQ6L1)',
	'MSTP(52)=2     ! work with LHAPDF',
        'MSTP(81)=1     ! multiple parton interactions 1 is Pythia default', 
        'MSTP(82)=4     ! Defines the multi-parton model', 
        'MSTU(21)=1     ! Check on possible errors during program execution', 
        'PARP(82)=1.8387   ! pt cutoff for multiparton interactions', 
        'PARP(89)=1960. ! sqrts for which PARP82 is set', 
        'PARP(83)=0.5   ! Multiple interactions: matter distrbn parameter', 
        'PARP(84)=0.4   ! Multiple interactions: matter distribution parameter', 
        'PARP(90)=0.16  ! Multiple interactions: rescaling power', 
        'PARP(67)=2.5    ! amount of initial-state radiation', 
        'PARP(85)=1.0  ! gluon prod. mechanism in MI', 
        'PARP(86)=1.0  ! gluon prod. mechanism in MI', 
        'PARP(62)=1.25   ! ', 
        'PARP(64)=0.2    ! ', 
        'MSTP(91)=1      !', 
        'PARP(91)=2.1   ! kt distribution', 
        'PARP(93)=15.0  ! '
    )
)
# foo bar baz
# l5SvXm8i8sKBo
# EyjywHdWBH1Jf
