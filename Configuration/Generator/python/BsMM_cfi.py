import FWCore.ParameterSet.Config as cms

source = cms.Source("PythiaSource",
    pythiaPylistVerbosity = cms.untracked.int32(0),
    filterEfficiency = cms.untracked.double(0.00045),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    crossSection = cms.untracked.double(54700000000.0),
    maxEventsToPrint = cms.untracked.int32(0),
    PythiaParameters = cms.PSet(
        pythiaUESettings = cms.vstring('MSTJ(11)=3     ! Choice of the fragmentation function', 'MSTJ(22)=2     ! Decay those unstable particles', 'PARJ(71)=10 .  ! for which ctau  10 mm', 'MSTP(2)=1      ! which order running alphaS', 'MSTP(33)=0     ! no K factors in hard cross sections', 'MSTP(51)=7     ! structure function chosen', 'MSTP(81)=1     ! multiple parton interactions 1 is Pythia default', 'MSTP(82)=4     ! Defines the multi-parton model', 'MSTU(21)=1     ! Check on possible errors during program execution', 'PARP(82)=1.9409   ! pt cutoff for multiparton interactions', 'PARP(89)=1960. ! sqrts for which PARP82 is set', 'PARP(83)=0.5   ! Multiple interactions: matter distrbn parameter', 'PARP(84)=0.4   ! Multiple interactions: matter distribution parameter', 'PARP(90)=0.16  ! Multiple interactions: rescaling power', 'PARP(67)=2.5    ! amount of initial-state radiation', 'PARP(85)=1.0  ! gluon prod. mechanism in MI', 'PARP(86)=1.0  ! gluon prod. mechanism in MI', 'PARP(62)=1.25   ! ', 'PARP(64)=0.2    ! ', 'MSTP(91)=1     !', 'PARP(91)=2.1   ! kt distribution', 'PARP(93)=15.0  ! '),
        parameterSets = cms.vstring('pythiaUESettings', 'processParameters'),
        processParameters = cms.vstring('PMAS(5,1)=4.8          ! b quark mass', 'MSEL=1                 ! Min Bias', 'MDME(953,2) = 0        ! PHASE SPACE', 'BRAT(953)   = 1.       ! BRANCHING FRACTION', 'KFDP(953,1) =  13      ! mu-', 'KFDP(953,2) = -13      ! mu+', 'KFDP(953,3) = 0        ! nada', 'KFDP(953,4) = 0        ! nada', 'KFDP(953,5) = 0        ! nada', 'PMAS(140,1) = 5.369', 'MDME(953,1) = 0        ', 'MDME(954,1) = 0        ', 'MDME(955,1) = 0        ', 'MDME(956,1) = 0        ', 'MDME(957,1) = 0        ', 'MDME(958,1) = 0        ', 'MDME(959,1) = 0        ', 'MDME(960,1) = 0        ', 'MDME(961,1) = 0        ', 'MDME(962,1) = 0        ', 'MDME(963,1) = 0        ', 'MDME(964,1) = 0        ', 'MDME(965,1) = 0        ', 'MDME(966,1) = 0        ', 'MDME(967,1) = 0        ', 'MDME(968,1) = 0        ', 'MDME(969,1) = 0        ', 'MDME(970,1) = 0        ', 'MDME(971,1) = 0        ', 'MDME(972,1) = 0        ', 'MDME(973,1) = 0        ', 'MDME(974,1) = 0        ', 'MDME(975,1) = 0        ', 'MDME(976,1) = 0        ', 'MDME(977,1) = 0        ', 'MDME(978,1) = 0        ', 'MDME(979,1) = 0        ', 'MDME(980,1) = 0        ', 'MDME(981,1) = 0        ', 'MDME(982,1) = 0        ', 'MDME(983,1) = 0        ', 'MDME(984,1) = 0        ', 'MDME(985,1) = 0        ', 'MDME(986,1) = 0        ', 'MDME(987,1) = 0        ', 'MDME(988,1) = 0        ', 'MDME(989,1) = 0        ', 'MDME(990,1) = 0        ', 'MDME(991,1) = 0        ', 'MDME(953,1) = 1       !  Bs -> mu mu ')
    )
)



