import FWCore.ParameterSet.Config as cms

pythiaUESettingsBlock = cms.PSet(
    pythiaUESettings = cms.vstring(
        'MSTU(21)=1     ! Check on possible errors during program execution', 
        'MSTJ(22)=2     ! Decay those unstable particles', 
        'PARJ(71)=10 .  ! for which ctau  10 mm', 
        'MSTP(2)=1      ! which order running alphaS', 
        'MSTP(33)=0     ! no K factors in hard cross sections', 
        'MSTP(51)=7     ! structure function chosen (internal PDF CTEQ5L)',
	'MSTP(52)=1     ! work with LHAPDF',
	'PARJ(1)=0.073  ! FLAV (Tuned by Professor on LEP data)',
	'PARJ(2)=0.2    ! FLAV (Tuned by Professor on LEP data)',
	'PARJ(3)=0.94   ! FLAV (Tuned by Professor on LEP data)',
	'PARJ(4)=0.032  ! FLAV (Tuned by Professor on LEP data)',
	'PARJ(11)=0.31  ! FLAV (Tuned by Professor on LEP data)',
	'PARJ(12)=0.4   ! FLAV (Tuned by Professor on LEP data)',
	'PARJ(13)=0.54  ! FLAV (Tuned by Professor on LEP data)',
	'PARJ(25)=0.63  ! FLAV (Tuned by Professor on LEP data)',
	'PARJ(26)=0.12  ! FLAV (Tuned by Professor on LEP data)',
        'MSTJ(11)=5     ! HAD Choice of the fragmentation function',    
	'PARJ(21)=0.313 ! HAD (Tuned by Professor on LEP data)', 
        'PARJ(41)=0.49  ! HAD (Tuned by Professor on LEP data)',                                     
        'PARJ(42)=1.2   ! HAD (Tuned by Professor on LEP data)',                                     
        'PARJ(46)=1.0   ! HAD (Tuned by Professor on LEP data)',                                     
        'PARJ(47)=1.0   ! HAD (Tuned by Professor on LEP data)',                                     
        'PARP(62)=2.9   ! ISR', 
        'PARP(64)=0.14  ! ISR', 
        'PARP(67)=2.65  ! ISR', 
        'MSTP(81)=1     ! MPI 1 is old Pythia set of models', 
        'MSTP(82)=4     ! MPI model', 
        'PARP(82)=1.9   ! MPI pt cutoff for multiparton interactions', 
        'PARP(83)=0.83  ! MPI matter distrbn parameter', 
        'PARP(84)=0.6   ! MPI matter distribution parameter', 
        'PARP(85)=0.86  ! MPI gluon prod. mechanism', 
        'PARP(86)=0.93  ! MPI gluon prod. mechanism', 
        'PARP(89)=1800. ! MPI sqrts for which PARP82 is set', 
        'PARP(90)=0.22  ! MPI rescaling power', 
        'MSTP(91)=1     ! BR', 
        'PARP(91)=2.1   ! BR kt distribution', 
        'PARP(93)=5.0   ! BR'
    )
)
# foo bar baz
# O7s4QIw0npDhy
