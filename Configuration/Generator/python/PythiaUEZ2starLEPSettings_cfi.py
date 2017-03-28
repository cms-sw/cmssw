import FWCore.ParameterSet.Config as cms

pythiaUESettingsBlock = cms.PSet(
        pythiaUESettings = cms.vstring(
                'MSTU(21)=1 ! Check on possible errors during program execution', 
                'MSTJ(22)=2 ! Decay those unstable particles', 
                'PARJ(71)=10 . ! for which ctau 10 mm', 
                'MSTP(33)=0 ! no K factors in hard cross sections', 
                'MSTP(2)=1 ! which order running alphaS', 
                'MSTP(51)=10042 ! structure function chosen (external PDF CTEQ6L1)', 
                'MSTP(52)=2 ! work with LHAPDF', 
                'PARP(82)=1.921 ! pt cutoff for multiparton interactions', 
                'PARP(89)=1800. ! sqrts for which PARP82 is set', 
                'PARP(90)=0.227 ! Multiple interactions: rescaling power', 
                'MSTP(95)=6 ! CR (color reconnection parameters)', 
                'PARP(77)=1.016 ! CR', 
                'PARP(78)=0.538 ! CR', 
                'PARP(80)=0.1 ! Prob. colored parton from BBR', 
                'PARP(83)=0.356 ! Multiple interactions: matter distribution parameter', 
                'PARP(84)=0.651 ! Multiple interactions: matter distribution parameter', 
                'PARP(62)=1.025 ! ISR cutoff', 
                'MSTP(91)=1 ! Gaussian primordial kT', 
                'PARP(93)=10.0 ! primordial kT-max', 
                'MSTP(81)=21 ! multiple parton interactions 1 is Pythia default', 
                'MSTP(82)=4 ! Defines the multi-parton model',
                #Z2star lep
                'PARJ(1) = 0.08 ! HAD diquark suppression',
                'PARJ(2) = 0.21 ! HAD strangeness suppression',
                'PARJ(3) = 0.94 ! HAD strange diquark suppression',
                'PARJ(4) = 0.04 ! HAD vectior diquark suppression',
                'PARJ(11) = 0.35 ! HAD P(vector meson), u and d only',
                'PARJ(12) = 0.35 ! HAD P(vector meson) contains ',
                'PARJ(13) = 0.54 ! HAD P(vector meson), heavy quarks',
                'PARJ(21) = 0.34 ! HAD fragmentation pt',
                'PARJ(25) = 0.63 ! HAD eta0 suppression',
                'PARJ(26) = 0.12 ! HAD eta0 suppression'
         ) 
)
