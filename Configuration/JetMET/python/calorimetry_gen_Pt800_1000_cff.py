import FWCore.ParameterSet.Config as cms

from GeneratorInterface.Pythia6Interface.pythiaDefault_cff import *
source = cms.Source("PythiaSource",
    pythiaVerbosity = cms.untracked.bool(False),
    PythiaParameters = cms.PSet(
        pythiaDefaultBlock,
        myParameters = cms.vstring('CKIN(3)=800.               !(D=0. GeV)  minimum pt hat for hard interactions', 
            'CKIN(4)=1000.               !(D=-1. GeV for no maximum) maximum pt hat for hard interactions', 
            'MSEL=1                     ! QCD hight pT processes (ISUB = 11, 12, 13, 28, 53, 68)', 
            'MSTJ(11)=3                 ! Choice of the fragmentation function', 
            'MSTJ(22)=2                 !Decay those unstable particles', 
            'MSTP(2)=1                  !which order running alphaS', 
            'MSTP(33)=0                 !(D=0) inclusion of K factors in (=0: none, i.e. K=1)', 
            'MSTP(51)=7                 !structure function chosen', 
            'MSTP(81)=1                 !multiple parton interactions 1 is Pythia default', 
            'MSTP(82)=4                 !Defines the multi-parton model', 
            'MSTU(21)=1                 !Check on possible errors during program execution', 
            'PARJ(71)=10.               !for which ctau  10 mm', 
            'PARP(82)=1.9               !pt cutoff for multiparton interactions', 
            'PARP(89)=1000.             !sqrts for which PARP82 is set', 
            'PARP(84)=0.4               !Multiple interactions: matter distribution Registered by Chris.Seez@cern.ch', 
            'PARP(90)=0.16              !Multiple interactions: rescaling power Registered by Chris.Seez@cern.ch', 
            'PMAS(5,1)=4.2              !mass of b quark', 
            'PMAS(6,1)=175.             !mass of top quark', 
            'PMAS(23,1)=91.187          !mass of Z', 
            'PMAS(24,1)=80.22           !mass of W'),
        parameterSets = cms.vstring('pythiaDefault', 
            'myParameters')
    )
)


