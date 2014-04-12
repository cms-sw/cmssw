import FWCore.ParameterSet.Config as cms

# Disclaimer :
#
# This set of Pythia configuration cards has been copied 
# and pasted from 
#   RecoTracker/RoadSearchCloudMaker/test/sim_pythia.cfg
# prepared by Oliver Gutsche (FNAL)
#
# the only change is MSTP(128)=2 - this option takes decay 
# products out of doc section; decay products point at parents
# in the main section (Julia Yarba, FNAL).
generator = cms.EDFilter("Pythia6GeneratorFilter",
    # to printout HepMC::GenEvent record (HepMC::GenEvent::print())
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    maxEventsToPrint = cms.untracked.int32(0),
    # to printout pythia event record (call pylist)
    pythiaPylistVerbosity = cms.untracked.int32(0),
    comEnergy = cms.double(8000.0),
    PythiaParameters = cms.PSet(
        pythiaHZZmumumumu = cms.vstring('PMAS(25,1)=190.0        !mass of Higgs', 
            'MSEL=0                  !(D=1) to select between full user control (0, then use MSUB) and some preprogrammed alternative: QCD hight pT processes (1, then ISUB=11, 12, 13, 28, 53, 68), QCD low pT processes (2, then ISUB=11, 12, 13, 28, 53, 68, 91, 92, 94, 95)', 
            'MSTJ(11)=3              !Choice of the fragmentation function', 
            'MSTJ(41)=1              !Switch off Pythia QED bremsshtrahlung', 
            'MSTP(51)=7              !structure function chosen', 
            'MSTP(61)=0              ! no initial-state showers', 
            'MSTP(71)=0              ! no final-state showers', 
            'MSTP(81)=0              ! no multiple interactions', 
            'MSTP(111)=0             ! no hadronization', 
            'MSTU(21)=1              !Check on possible errors during program execution', 
            'MSUB(102)=1             !ggH', 
            'MSUB(123)=1             !ZZ fusion to H', 
            'MSUB(124)=1             !WW fusion to H', 
            'PARP(82)=1.9            !pt cutoff for multiparton interactions', 
            'PARP(83)=0.5            !Multiple interactions: matter distrbn parameter Registered by Chris.Seez@cern.ch', 
            'PARP(84)=0.4            !Multiple interactions: matter distribution parameter Registered by Chris.Seez@cern.ch', 
            'PARP(90)=0.16           !Multiple interactions: rescaling power Registered by Chris.Seez@cern.ch', 
            'CKIN(45)=5.             !high mass cut on m2 in 2 to 2 process Registered by Chris.Seez@cern.ch', 
            'CKIN(46)=150.           !high mass cut on secondary resonance m1 in 2->1->2 process Registered by Alexandre.Nikitenko@cern.ch', 
            'CKIN(47)=5.             !low mass cut on secondary resonance m2 in 2->1->2 process Registered by Alexandre.Nikitenko@cern.ch', 
            'CKIN(48)=150.           !high mass cut on secondary resonance m2 in 2->1->2 process Registered by Alexandre.Nikitenko@cern.ch', 
            'MDME(174,1)=0           !Z decay into d dbar', 
            'MDME(175,1)=0           !Z decay into u ubar', 
            'MDME(176,1)=0           !Z decay into s sbar', 
            'MDME(177,1)=0           !Z decay into c cbar', 
            'MDME(178,1)=0           !Z decay into b bbar', 
            'MDME(179,1)=0           !Z decay into t tbar', 
            'MDME(182,1)=0           !Z decay into e- e+', 
            'MDME(183,1)=0           !Z decay into nu_e nu_ebar', 
            'MDME(184,1)=1           !Z decay into mu- mu+', 
            'MDME(185,1)=0           !Z decay into nu_mu nu_mubar', 
            'MDME(186,1)=0           !Z decay into tau- tau+', 
            'MDME(187,1)=0           !Z decay into nu_tau nu_taubar', 
            'MDME(210,1)=0           !Higgs decay into dd', 
            'MDME(211,1)=0           !Higgs decay into uu', 
            'MDME(212,1)=0           !Higgs decay into ss', 
            'MDME(213,1)=0           !Higgs decay into cc', 
            'MDME(214,1)=0           !Higgs decay into bb', 
            'MDME(215,1)=0           !Higgs decay into tt', 
            'MDME(216,1)=0           !Higgs decay into', 
            'MDME(217,1)=0           !Higgs decay into Higgs decay', 
            'MDME(218,1)=0           !Higgs decay into e nu e', 
            'MDME(219,1)=0           !Higgs decay into mu nu mu', 
            'MDME(220,1)=0           !Higgs decay into tau nu tau', 
            'MDME(221,1)=0           !Higgs decay into Higgs decay', 
            'MDME(222,1)=0           !Higgs decay into g g', 
            'MDME(223,1)=0           !Higgs decay into gam gam', 
            'MDME(224,1)=0           !Higgs decay into gam Z', 
            'MDME(225,1)=1           !Higgs decay into Z Z', 
            'MDME(226,1)=0           !Higgs decay into W W', 
            'MSTP(128)=2             !dec.prods out of doc section, point at parents in the main section'),
        parameterSets = cms.vstring('pythiaHZZmumumumu')
    )
)
