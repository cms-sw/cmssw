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

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(5000))

process.source = cms.Source("EmptySource")

from Configuration.Generator.PythiaUESettings_cfi import *

process.generator = cms.EDFilter("Pythia6GeneratorFilter",
    pythiaHepMCVerbosity = cms.untracked.bool(True),
    maxEventsToPrint = cms.untracked.int32(5),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    comEnergy = cms.double(7000.0),
    ExternalDecays = cms.PSet(
        EvtGen = cms.untracked.PSet(
             operates_on_particles = cms.vint32( 0 ), # 0 (zero) means default list (hardcoded)
                                                      # you can put here the list of particles (PDG IDs)
                                                      # that you want decayed by EvtGen
	     use_default_decay = cms.untracked.bool(False),
             decay_table = cms.FileInPath('GeneratorInterface/ExternalDecays/data/DECAY_NOLONGLIFE.DEC'),
             # decay_table = cms.FileInPath('GeneratorInterface/ExternalDecays/data/DECAY.DEC'),
             particle_property_file = cms.FileInPath('GeneratorInterface/ExternalDecays/data/evt.pdl'),
             user_decay_file = cms.FileInPath('GeneratorInterface/ExternalDecays/data/Bs_DsStarlnu_DsGamma.dec'),
             # user_decay_file = cms.FileInPath('GeneratorInterface/ExternalDecays/data/Bs_Jpsiphi_mumuKK.dec'),
             list_forced_decays = cms.vstring('MyB_s0','Myanti-B_s0'),
             ),
        parameterSets = cms.vstring('EvtGen')
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
