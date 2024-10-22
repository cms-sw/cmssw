import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
#process.load("SimGeneral.HepPDTESSource.pdt_cfi")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    generator = cms.PSet(
        initialSeed = cms.untracked.uint32(123456789),
        engineName = cms.untracked.string('HepJamesRandom')
    )
)


# The following three lines reduce the clutter of repeated printouts
# of the same exception message.
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.MessageLogger.cerr.enableStatistics = False


process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(20))

###process.source = cms.Source("EmptySource")
process.source = cms.Source("EmptySource",
     firstLuminosityBlock = cms.untracked.uint32(1),
     numberEventsInLuminosityBlock = cms.untracked.uint32(5),
     firstEvent = cms.untracked.uint32(1),
     firstRun = cms.untracked.uint32(1),
     numberEventsInRun = cms.untracked.uint32(10)
)

process.generator = cms.EDFilter("Pythia6GeneratorFilter",
    pythiaHepMCVerbosity = cms.untracked.bool(True),
    maxEventsToPrint = cms.untracked.int32(3),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    # this shows how to turn ON some of the general Py6 printouts, like banner...
    # --> displayPythiaBanner = cms.untracked.bool(True),
    displayPythiaCards = cms.untracked.bool(True),
    ElectronProtonInitialState = cms.PSet(
       electronMomentum = cms.double(400.),
       protonMomentum   = cms.double(-400.)
    ),
    comEnergy = cms.double(0.0), # set to 0.0 as it's NOT used with this initial state 
                                 # the parameter but must be present (by Fwk)
    PythiaParameters = cms.PSet(

        pythiaHZZ4tau = cms.vstring('PMAS(25,1)=190.0        !mass of Higgs', 
            'MSEL=0                  !(D=1) to select between full user control (0, then use MSUB) and some preprogrammed alternative: QCD hight pT processes (1, then ISUB=11, 12, 13, 28, 53, 68), QCD low pT processes (2, then ISUB=11, 12, 13, 28, 53, 68, 91, 92, 94, 95)', 
            'MSTJ(11)=3              !Choice of the fragmentation function', 
            'MSTJ(41)=3              !Switch off Pythia QED bremsshtrahlung off leptons (only)', 
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
            'MDME(184,1)=0           !Z decay into mu- mu+', 
            'MDME(185,1)=0           !Z decay into nu_mu nu_mubar', 
            'MDME(186,1)=1           !Z decay into tau- tau+', 
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
            'MSTP(128)=0             !dec.prods out of doc section, point at parents in the main section'),

        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring( 
            'pythiaHZZ4tau')
    )
)

process.GEN = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('Py6_HZZ4tau.root')
)

process.p = cms.Path(process.generator)
process.outpath = cms.EndPath(process.GEN)

process.schedule = cms.Schedule(process.p, process.outpath)
