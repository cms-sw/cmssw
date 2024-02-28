import FWCore.ParameterSet.Config as cms

cepgenOutputModules = cms.untracked.PSet(
    # place here all CepGen modules steering for output
    # e.g. for a printout of every 1000th event:
    #dump = cms.PSet(printEvery = cms.uint32(1000))
)

cepgenPythia6BeamFragmenter = cms.untracked.PSet(
    pythia6 = cms.PSet(
        maxTrials = cms.int32(10),
        seed = cms.int32(42),
        preConfiguration = cms.vstring(
            'MSTU(21)=1     ! Check on possible errors during program execution',
            'MSTU(25)=0     ! No warnings are written',
            'MSTJ(22)=2     ! Decay those unstable particles',
            'PARJ(71)=10 .  ! for which ctau  10 mm',
            'MSTP(33)=0     ! no K factors in hard cross sections',
            'MSTP(2)=1      ! which order running alphaS',
            'MSTP(81)=0     ! multiple parton interactions 1 is Pythia default'
        )
    )
)
