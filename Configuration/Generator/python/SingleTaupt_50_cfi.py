import FWCore.ParameterSet.Config as cms

source = cms.Source("EmptySource")

generator = cms.EDProducer("Pythia6PtGun",
    PGunParameters = cms.untracked.PSet(
        ParticleID = cms.untracked.vint32(-15),
        AddAntiParticle = cms.untracked.bool(False),
        MinPhi = cms.untracked.double(0),
        MaxPhi = cms.untracked.double(360),
        MinPt = cms.untracked.double(50.0),
        MaxPt = cms.untracked.double(50.0001),
        MinEta = cms.untracked.double(-2.4),
        MaxEta = cms.untracked.double(2.4)
    ),
    PythiaParameters = cms.PSet(
        pythiaTauJets = cms.vstring(
            'MDME(89,1)=0      ! no tau->electron',
            'MDME(90,1)=0      ! no tau->muon'
        ),
        pythiaUESettings = cms.vstring(
            'MSTJ(11)=3     ! Choice of the fragmentation function',
            'MSTJ(22)=2     ! Decay those unstable particles',
            'PARJ(71)=10 .  ! for which ctau  10 mm',
            'MSTP(2)=1      ! which order running alphaS',
            'MSTP(33)=0     ! no K factors in hard cross sections',
            'MSTP(51)=7     ! structure function chosen',
            'MSTP(81)=1     ! multiple parton interactions 1 is Pythia default',
            'MSTP(82)=4     ! Defines the multi-parton model',
            'MSTU(21)=1     ! Check on possible errors during program execution',
            'PARP(82)=1.9409   ! pt cutoff for multiparton interactions',
            'PARP(89)=1960. ! sqrts for which PARP82 is set',
            'PARP(83)=0.5   ! Multiple interactions: matter distrbn parameter',
            'PARP(84)=0.4   ! Multiple interactions: matter distribution parameter',
            'PARP(90)=0.16  ! Multiple interactions: rescaling power',
            'PARP(67)=2.5    ! amount of initial-state radiation',
            'PARP(85)=1.0  ! gluon prod. mechanism in MI',
            'PARP(86)=1.0  ! gluon prod. mechanism in MI',
            'PARP(62)=1.25   ! ',
            'PARP(64)=0.2    ! ',
            'MSTP(91)=1     !',
            'PARP(91)=2.1   ! kt distribution',
            'PARP(93)=15.0  ! '
        ),
        parameterSets = cms.vstring(
            'pythiaUESettings',
            'pythiaTauJets'
        )
    )
)

ProductionFilterSequence = cms.Sequence(generator)
