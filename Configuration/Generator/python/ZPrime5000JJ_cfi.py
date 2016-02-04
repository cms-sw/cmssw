import FWCore.ParameterSet.Config as cms

from GeneratorInterface.Pythia6Interface.pythiaDefault_cff import *
generator = cms.EDFilter("Pythia6GeneratorFilter",
    pythiaVerbosity = cms.untracked.bool(False),
    comEnergy = cms.double(10000.0),
    PythiaParameters = cms.PSet(
        # Default (mostly empty - to keep PYTHIA default) card file
        # Name of the set is "pythiaDefault"
        pythiaDefaultBlock,
        # User cards - name is "myParameters"
        myParameters = cms.vstring('PMAS(32,1)= 5000.            !mass of Zprime', 
            'MSEL=0                      !(D=1) to select between full user control (0, then use MSUB) and some preprogrammed alternative', 
            'MSTP(44) = 3                !only select the Z process', 
            'MSUB(141) = 1               !ff  gamma z0 Z0', 
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
            'PMAS(24,1)=80.22           !mass of W', 
            'MDME(289,1)= 1            !d dbar', 
            'MDME(290,1)= 1            !u ubar', 
            'MDME(291,1)= 1            !s sbar', 
            'MDME(292,1)= 1            !c cbar', 
            'MDME(293,1)= 0            !b bar', 
            'MDME(294,1)= 0            !t tbar', 
            'MDME(295,1)= 0            !4th gen Q Qbar', 
            'MDME(296,1)= 0            !4th gen Q Qbar', 
            'MDME(297,1)= 0            !e e', 
            'MDME(298,1)= 0            !neutrino e e', 
            'MDME(299,1)= 0            ! mu mu', 
            'MDME(300,1)= 0            !neutrino mu mu', 
            'MDME(301,1)= 0            !tau tau', 
            'MDME(302,1)= 0            !neutrino tau tau', 
            'MDME(303,1)= 0            !4th generation lepton', 
            'MDME(304,1)= 0            !4th generation neutrino', 
            'MDME(305,1)= 0            !W W', 
            'MDME(306,1)= 0            !H  charged higgs', 
            'MDME(307,1)= 0            !Z', 
            'MDME(308,1)= 0            !Z', 
            'MDME(309,1)= 0            !sm higgs', 
            'MDME(310,1)= 0            !weird neutral higgs HA'),
        # This is a vector of ParameterSet names to be read, in this order
        # The first two are in the include files below
        # The last one are simply my additional parameters
        parameterSets = cms.vstring('pythiaDefault', 
            'myParameters')
    )
)
