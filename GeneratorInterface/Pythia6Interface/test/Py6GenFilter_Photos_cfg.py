import FWCore.ParameterSet.Config as cms

#
# WARNING: This is NOT an example for users -
#          it's my private (JY) "development" cfg, for testing
#          newly implemented PhotosInterface - which is NOT yet
#          released via ExternalDecayDeriver
#

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




process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(20))

process.source = cms.Source("EmptySource")

process.generator = cms.EDFilter("Pythia6GeneratorFilter",
    pythiaHepMCVerbosity = cms.untracked.bool(True),
    maxEventsToPrint = cms.untracked.int32(3),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    # this shows how to turn ON some of the general Py6 printouts, like banner...
    ## --> displayPythiaBanner = cms.untracked.bool(True),
    ## --> displayPythiaCards = cms.untracked.bool(True),
    comEnergy = cms.double(10000.0),

    ExternalDecays = cms.PSet(
        Photos = cms.untracked.PSet(),
        parameterSets = cms.vstring( "Photos" )
    ),

    PythiaParameters = cms.PSet(

        pythiaSimpleSettings = cms.vstring(
	   'PMAS(5,1)=4.8 ! b quark mass',
           'PMAS(6,1)=172.3 ! t quark mass',
           'MSTP(61)=0 ! turn off initial state radiation',
           'mstj(41)=1' # per Steve M., instead of mstp(71)... btw, shoult it be 0 or 1 ?
           #'MSTP(71)=0 ! final state radiation (or final-state showers ?)'

	),
	pythiaSpecialSettings = cms.vstring(
	    'PMAS(25,1)=190.0        !mass of Higgs', 
            'MSEL=0                  !(D=1) to select between full user control (0, then use MSUB) and some preprogrammed alternative: QCD hight pT processes (1, then ISUB=11, 12, 13, 28, 53, 68), QCD low pT processes (2, then ISUB=11, 12, 13, 28, 53, 68, 91, 92, 94, 95)', 
            'MSTJ(11)=3              !Choice of the fragmentation function', 
            'MSTJ(41)=1              !Switch off Pythia QED bremsshtrahlung', 
            'MSTP(51)=7              !structure function chosen', 
            'MSTP(61)=0              ! no initial-state showers', 
            'MSTP(71)=0              ! no final-state showers', 
            'MSTP(81)=0              ! no multiple interactions', 
            'MSTP(111)=0             ! no hadronization', 
            'MSTU(21)=1              !Check on possible errors during program execution', 
	    # these 4 below are irrelevant if the hadronization is off (mstp(111)=0)
            'PARP(82)=1.9            !pt cutoff for multiparton interactions', 
            'PARP(83)=0.5            !Multiple interactions: matter distrbn parameter Registered by Chris.Seez@cern.ch', 
            'PARP(84)=0.4            !Multiple interactions: matter distribution parameter Registered by Chris.Seez@cern.ch', 
            'PARP(90)=0.16           !Multiple interactions: rescaling power Registered by Chris.Seez@cern.ch', 
	    #........................................
            'CKIN(45)=5.             !high mass cut on m2 in 2 to 2 process Registered by Chris.Seez@cern.ch', 
            'CKIN(46)=150.           !high mass cut on secondary resonance m1 in 2->1->2 process Registered by Alexandre.Nikitenko@cern.ch', 
            'CKIN(47)=5.             !low mass cut on secondary resonance m2 in 2->1->2 process Registered by Alexandre.Nikitenko@cern.ch', 
            'CKIN(48)=150.           !high mass cut on secondary resonance m2 in 2->1->2 process Registered by Alexandre.Nikitenko@cern.ch'
	),
	pythiaProcessSettings = cms.vstring(
            'MSUB(102)=1             !ggH', 
            'MSUB(123)=1             !ZZ fusion to H', 
            'MSUB(124)=1             !WW fusion to H', 
	),
	pythiaHZZDecays = cms.vstring(
            'MDME(174,1)=0           !Z decay into d dbar', 
            'MDME(175,1)=0           !Z decay into u ubar', 
            'MDME(176,1)=0           !Z decay into s sbar', 
            'MDME(177,1)=0           !Z decay into c cbar', 
            # set it to 4 for the 1st Z to go to b bbar...
	    'MDME(178,1)=0           !Z decay into b bbar',
	    # ............................................ 
            'MDME(179,1)=0           !Z decay into t tbar', 
            'MDME(182,1)=0           !Z decay into e- e+', 
            'MDME(183,1)=0           !Z decay into nu_e nu_ebar', 
            'MDME(184,1)=4           !Z decay into mu- mu+', 
            'MDME(185,1)=0           !Z decay into nu_mu nu_mubar', 
	    # ... and this one to 5, for the 2nd Z to goe to tau- tau+
            'MDME(186,1)=5           !Z decay into tau- tau+', 
	    # ............................................
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
            'MDME(226,1)=0           !Higgs decay into W W'
        ),
        pythiaTauL = cms.vstring(
           "mdme(89,1)=1", # tau -> e
           "mdme(90,1)=1", # tau -> mu
           # all other tau decays OFF
           "mdme(91,1)=0",
           "mdme(92,1)=0",
           "mdme(93,1)=0",
           "mdme(94,1)=0",
           "mdme(95,1)=0",
           "mdme(96,1)=0",
           "mdme(97,1)=0",
           "mdme(98,1)=0",
           "mdme(99,1)=0",
           "mdme(100,1)=0",
           "mdme(101,1)=0",
           "mdme(102,1)=0",
           "mdme(103,1)=0",
           "mdme(104,1)=0",
           "mdme(105,1)=0",
           "mdme(106,1)=0",
           "mdme(107,1)=0",
           "mdme(108,1)=0",
           "mdme(109,1)=0",
           "mdme(110,1)=0",
           "mdme(111,1)=0",
           "mdme(112,1)=0",
           "mdme(113,1)=0",
           "mdme(114,1)=0",
           "mdme(115,1)=0",
           "mdme(116,1)=0",
           "mdme(117,1)=0",
           "mdme(118,1)=0",
           "mdme(119,1)=0",
           "mdme(120,1)=0",
           "mdme(121,1)=0",
           "mdme(122,1)=0",
           "mdme(123,1)=0",
           "mdme(124,1)=0",
           "mdme(125,1)=0",
           "mdme(126,1)=0",
           "mdme(127,1)=0",
           "mdme(128,1)=0",
           "mdme(129,1)=0",
           "mdme(130,1)=0",
           "mdme(131,1)=0",
           "mdme(132,1)=0",
           "mdme(133,1)=0",
           "mdme(134,1)=0",
           "mdme(135,1)=0",
           "mdme(136,1)=0",
           "mdme(137,1)=0",
           "mdme(138,1)=0",
           "mdme(139,1)=0",
           "mdme(140,1)=0",
           "mdme(141,1)=0",
           "mdme(142,1)=0"
        ),
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring( 
	    'pythiaSpecialSettings',
	    ##'pythiaSimpleSettings', 
	    'pythiaProcessSettings',
            'pythiaHZZDecays', 'pythiaTauL' )
    )
)

process.GEN = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('TestHZZleptons.root')
)

process.p = cms.Path(process.generator)
process.outpath = cms.EndPath(process.GEN)

process.schedule = cms.Schedule(process.p, process.outpath)
