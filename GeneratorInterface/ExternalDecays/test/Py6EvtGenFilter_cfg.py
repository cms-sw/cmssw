import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")
process.load("Configuration.Generator.PythiaUESettings_cfi")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    generator = cms.PSet(
        initialSeed = cms.untracked.uint32(123456789),
        engineName = cms.untracked.string('HepJamesRandom')
    )
)

# The following three lines reduce the clutter of repeated printouts
# of the same exception message.
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.destinations = ['cerr']
process.MessageLogger.statistics = []
process.MessageLogger.fwkJobReports = []

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10000))

process.source = cms.Source("EmptySource")

from Configuration.Generator.PythiaUESettings_cfi import *

process.generator = cms.EDFilter("Pythia6GeneratorFilter",
    pythiaHepMCVerbosity = cms.untracked.bool(True),
    maxEventsToPrint = cms.untracked.int32(2),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    comEnergy = cms.double(10000.0),
    ExternalDecays = cms.PSet(
        EvtGen = cms.untracked.PSet(
	     use_default_decay = cms.untracked.bool(False),
             decay_table = cms.FileInPath('GeneratorInterface/EvtGenInterface/data/DECAY.DEC'),
             particle_property_file = cms.FileInPath('GeneratorInterface/EvtGenInterface/data/evt.pdl'),
             user_decay_file = cms.FileInPath('GeneratorInterface/EvtGenInterface/data/Validation.dec'),
             list_forced_decays = cms.vstring('MyB0','Myanti-B0','MyB_s0','Myanti-B_s0'),
             processParameters = cms.vstring('MDCY(134,1) = 0', 
                   'MDCY(137,1) = 0', 
                   'MDCY(138,1) = 0', 
                   'MDCY(135,1) = 0', 
                   'MDCY(141,1) = 0', 
                   'MDCY(140,1) = 0', 
                   'MDCY(15,1) = 0', 
                   'MDCY(123,1) = 0', 
                   'MDCY(126,1) = 0', 
                   'MDCY(129,1) = 0', 
                   'MDCY(122,1) = 0', 
                   'MDCY(125,1) = 0', 
                   'MDCY(128,1) = 0', 
                   'MDCY(262,1) = 0', 
                   'MDCY(264,1) = 0', 
                   'MDCY(263,1) = 0', 
                   'MDCY(265,1) = 0', 
                   'MDCY(286,1) = 0', 
                   'MDCY(287,1) = 0', 
                   'MDCY(124,1) = 0', 
                   'MDCY(127,1) = 0', 
                   'MDCY(266,1) = 0', 
                   'MDCY(288,1) = 0', 
                   'MDCY(267,1) = 0', 
                   'MDCY(130,1) = 0', 
                   'MDCY(112,1) = 0', 
                   'MDCY(113,1) = 0', 
                   'MDCY(114,1) = 0', 
                   'MDCY(117,1) = 0', 
                   'MDCY(258,1) = 0', 
                   'MDCY(256,1) = 0', 
                   'MDCY(257,1) = 0', 
                   'MDCY(259,1) = 0', 
                   'MDCY(284,1) = 0', 
                   'MDCY(283,1) = 0', 
                   'MDCY(118,1) = 0', 
                   'MDCY(115,1) = 0', 
                   'MDCY(102,1) = 0', 
                   'MDCY(109,1) = 0', 
                   'MDCY(103,1) = 0', 
                   'MDCY(107,1) = 0', 
                   'MDCY(110,1) = 0', 
                   'MDCY(119,1) = 0', 
                   'MDCY(120,1) = 0', 
                   'MDCY(281,1) = 0', 
                   'MDCY(280,1) = 0', 
                   'MDCY(281,1) = 0', 
                   'MDCY(108,1) = 0', 
                   'MDCY(104,1) = 0', 
                   'MDCY(253,1) = 0', 
                   'MDCY(251,1) = 0', 
                   'MDCY(250,1) = 0', 
                   'MDCY(252,1) = 0', 
                   'MDCY(254,1) = 0', 
                   'MDCY(282,1) = 0', 
                   'MDCY(285,1) = 0', 
                   'MDCY(111,1) = 0', 
                   'MDCY(121,1) = 0', 
                   'MDCY(255,1) = 0', 
                   'MDCY(261,1) = 0', 
                   'MDCY(131,1) = 0', 
                   'MDCY(132,1) = 0', 
                   'MDCY(295,1) = 0', 
                   'MDCY(268,1) = 0', 
                   'MDCY(289,1) = 0', 
                   'MDCY(133,1) = 0', 
                   'MDCY(146,1) = 0', 
                   'MDCY(147,1) = 0', 
                   'MDCY(296,1) = 0', 
                   'MDCY(278,1) = 0', 
                   'MDCY(294,1) = 0', 
                   'MDCY(148,1) = 0', 
                   'MDCY(279,1) = 0', 
                   'MDCY(181,1) = 0', 
                   'MDCY(182,1) = 0', 
                   'MDCY(84,1) = 0', 
                   'MDCY(179,1) = 0', 
                   'MDCY(185,1) = 0', 
                   'MDCY(189,1) = 0', 
                   'MDCY(187,1) = 0', 
                   'MDCY(194,1) = 0', 
                   'MDCY(192,1) = 0', 
                   'MDCY(164,1) = 0', 
                   'MDCY(169,1) = 0', 
                   'MDCY(158,1) = 0', 
                   'MDCY(159,1) = 0', 
                   'MDCY(175,1) = 0', 
                   'MDCY(155,1) = 0', 
                   'MDCY(151,1) = 0', 
                   'MDCY(162,1) = 0', 
                   'MDCY(167,1) = 0', 
                   'MDCY(163,1) = 0', 
                   'MDCY(170,1) = 0', 
                   'MDCY(168,1) = 0', 
                   'MDCY(174,1) = 0', 
                   'MDCY(172,1) = 0', 
                   'MDCY(173,1) = 0', 
                   'MDCY(176,1) = 0', 
                   'MDCY(180,1) = 0', 
                   'MDCY(186,1) = 0', 
                   'MDCY(188,1) = 0', 
                   'MDCY(193,1) = 0', 
                   'MDCY(195,1) = 0', 
                   'MDCY(196,1) = 0', 
                   'MDCY(197,1) = 0', 
                   'MDCY(43,1) = 0', 
                   'MDCY(44,1) = 0', 
                   'MDCY(269,1) = 0', 
                   'MDCY(210,1) = 0', 
                   'MDCY(211,1) = 0', 
                   'MDCY(219,1) = 0', 
                   'MDCY(227,1) = 0', 
                   'MDCY(217,1) = 0', 
                   'MDCY(208,1) = 0', 
                   'MDCY(215,1) = 0', 
                   'MDCY(143,1) = 0', 
                   'MDCY(223,1) = 0', 
                   'MDCY(225,1) = 0', 
                   'MDCY(272,1) = 0', 
                   'MDCY(291,1) = 0', 
                   'MDCY(273,1) = 0', 
                   'MDCY(139,1) = 0', 
                   'MDCY(270,1) = 0', 
                   'MDCY(290,1) = 0', 
                   'MDCY(271,1) = 0', 
                   'MDCY(136,1) = 0', 
                   'MDCY(274,1) = 0', 
                   'MDCY(292,1) = 0', 
                   'MDCY(275,1) = 0', 
                   'MDCY(142,1) = 0', 
                   'MDCY(144,1) = 0', 
                   'MDCY(145,1) = 0', 
                   'MDCY(209,1) = 0', 
                   'MDCY(218,1) = 0', 
                   'MDCY(216,1) = 0', 
                   'MDCY(224,1) = 0', 
                   'MDCY(226,1) = 0', 
                   'MDCY(228,1) = 0', 
                   'MDCY(276,1) = 0', 
                   'MDCY(277,1) = 0', 
                   'MDCY(293,1) = 0', 
                   'MDCY(105,1) = 0')
             ),
        parameterSets = cms.vstring('EvtGen')
    ),
    PythiaParameters = cms.PSet(

        process.pythiaUESettingsBlock,
        bbbarSettings = cms.vstring('MSEL = 5'), 
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring('pythiaUESettings','bbbarSettings')
    )
)

process.GEN = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('TestEvtGen.root')
)

process.p = cms.Path(process.generator)
process.outpath = cms.EndPath(process.GEN)

process.schedule = cms.Schedule(process.p, process.outpath)
