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

process.MessageLogger.cerr.enableStatistics = False


process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(500))

process.source = cms.Source("EmptySource")

from Configuration.Generator.PythiaUESettings_cfi import *

process.generator = cms.EDFilter("Pythia6GeneratorFilter",
    pythiaHepMCVerbosity = cms.untracked.bool(True),
    maxEventsToPrint = cms.untracked.int32(5),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    comEnergy = cms.double(7000.0),
    ExternalDecays = cms.PSet(
        EvtGen1 = cms.untracked.PSet(
             decay_table = cms.string('GeneratorInterface/EvtGenInterface/data/DECAY_2010.DEC'),
             particle_property_file = cms.FileInPath('GeneratorInterface/EvtGenInterface/data/evt.pdl'),
             #user_decay_files = cms.vstring('DECAY_2010.DEC'),
             user_decay_file = cms.vstring('GeneratorInterface/ExternalDecays/data/LambdaB_JPsiLambda_ppi.dec'),
             list_forced_decays = cms.vstring('MyLambda_b0','Myanti-Lambda_b0'),
             particles_to_polarize = cms.untracked.vint32(5122, -5122),
             particle_polarizations = cms.untracked.vdouble(-0.4, -0.4),
             operates_on_particles = cms.vint32(0), # 0 (zero) means default list (hardcoded), the list of PDG IDs can be put here
	     use_default_decay = cms.untracked.bool(False)
             # decay_table = cms.FileInPath('GeneratorInterface/ExternalDecays/data/DECAY_NOLONGLIFE.DEC')
             # decay_table = cms.FileInPath('GeneratorInterface/ExternalDecays/data/DECAY.DEC')
             # user_decay_file = cms.FileInPath('GeneratorInterface/ExternalDecays/data/Bs_DsStarlnu_DsGamma.dec')
             ),
        parameterSets = cms.vstring('EvtGen1')
    ),
    PythiaParameters = cms.PSet(

        process.pythiaUESettingsBlock,
        bbbarSettings = cms.vstring('MSEL=5          ! bbbar '),
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring('pythiaUESettings','bbbarSettings')
    )
)

process.GEN = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('/tmp/TestEvtGen.root')
)

process.p = cms.Path(process.generator)
process.outpath = cms.EndPath(process.GEN)

process.schedule = cms.Schedule(process.p, process.outpath)
