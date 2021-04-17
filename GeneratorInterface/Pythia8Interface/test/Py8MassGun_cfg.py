import FWCore.ParameterSet.Config as cms

from Configuration.Generator.Pythia8CommonSettings_cfi import *

process = cms.Process("TEST")

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.source = cms.Source("EmptySource")

process.generator = cms.EDFilter("Pythia8MassGun", 
    maxEventsToPrint = cms.untracked.int32(100),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    pythiaHepMCVerbosity = cms.untracked.bool(True),

    PGunParameters = cms.PSet(
       ParticleID = cms.vint32(999999),
        # this defines "absolute" energy spead of particles in the jet
	MinM   = cms.double(8.0),
	MaxM   = cms.double(15.0),
	# the following params define the boost
        MinP   = cms.double(20.0),
        MaxP   = cms.double(20.0),
        MomMode = cms.int32(1),
        MinPt   = cms.double(20.0),
        MaxPt   = cms.double(20.0),
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double(3.14159265359),
	MinEta = cms.double(-2.4),
        MaxEta = cms.double(2.4)
    ),
    PythiaParameters = cms.PSet(

    pythia8CommonSettingsBlock,

    processParameters = cms.vstring(
      '999999:all = GeneralResonance void 1 0 0 500. 1. 0. 0. 0.',
      '999999:oneChannel = 1 1.00 101 15 -15 15 -15',
      'Main:timesAllowErrors    = 10000',
      '15:onMode = off',
      '15:onIfAll = 211 211 211',
      '15:onIfAll = 211 211 321',
      '15:onIfAll = 211 321 321',
      '15:onIfAll = 321 321 321',
      '15:onIfAll = 321 321 321',
      ),

    parameterSets = cms.vstring(
        'processParameters',
	'pythia8CommonSettings')
    )
)

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(2)
        ),
        enable = cms.untracked.bool(True)
    )
)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    generator = cms.PSet(
        initialSeed = cms.untracked.uint32(123456789),
        engineName = cms.untracked.string('HepJamesRandom')
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

process.GEN = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('Py8JetGun.root')
)


process.p = cms.Path(process.generator)
process.outpath = cms.EndPath(process.GEN)

process.schedule = cms.Schedule(process.p, process.outpath)

