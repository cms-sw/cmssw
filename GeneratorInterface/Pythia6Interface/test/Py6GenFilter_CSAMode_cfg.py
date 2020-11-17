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


process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(5))

process.source = cms.Source("EmptySource")

from Configuration.Generator.PythiaUESettings_cfi import *

process.generator = cms.EDFilter("Pythia6GeneratorFilter",
    pythiaHepMCVerbosity = cms.untracked.bool(True),
    maxEventsToPrint = cms.untracked.int32(0),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    comEnergy = cms.double(10000.0),
    PythiaParameters = cms.PSet(
        pythiaUESettingsBlock,
        processParameters = cms.vstring('MSEL=1         ! Quarkonia', 
	    'CKIN(3)=50.',
            'MSTP(142)=2      ! turns on the PYEVWT Pt re-weighting routine'),
	CSAParameters = cms.vstring('CSAMODE = 7 ! cross-section reweighted quarkonia',
	                            'PTPOWER = 5 ! pt^6'),
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring('pythiaUESettings', 
            'processParameters',
	    'CSAParameters')
    )
)

process.GEN = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('TestCSAMode.root')
)

process.p = cms.Path(process.generator)
process.outpath = cms.EndPath(process.GEN)

process.schedule = cms.Schedule(process.p, process.outpath)
