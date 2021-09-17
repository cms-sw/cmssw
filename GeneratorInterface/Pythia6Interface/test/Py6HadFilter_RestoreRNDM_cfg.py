import FWCore.ParameterSet.Config as cms

from Configuration.Generator.PythiaUESettings_cfi import *

process = cms.Process("RERUN")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
#process.load("SimGeneral.HepPDTESSource.pdt_cfi")

process.load("Configuration.StandardSequences.Services_cff")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    generator = cms.PSet(
        initialSeed = cms.untracked.uint32(123456789),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    restoreStateLabel = cms.untracked.string('randomEngineStateProducer')
)

## process.randomEngineStateProducer = cms.EDProducer("RandomEngineStateProducer")

# The following three lines reduce the clutter of repeated printouts
# of the same exception message.
process.load("FWCore.MessageLogger.MessageLogger_cfi")




process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(5))

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:TestTTbar.root'),
    skipEvents = cms.untracked.uint32(3)
)

process.generator = cms.EDFilter("Pythia6HadronizerFilter",
    pythiaHepMCVerbosity = cms.untracked.bool(True),
    maxEventsToPrint = cms.untracked.int32(5),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    comEnergy = cms.double(10000.0),
    PythiaParameters = cms.PSet(
        pythiaUESettingsBlock,
        processParameters = cms.vstring('MSEL=0         ! User defined processes', 
                        'PMAS(5,1)=4.4   ! b quark mass',
                        'PMAS(6,1)=172.4 ! t quark mass',
			'MSTJ(1)=1       ! Fragmentation/hadronization on or off',
			'MSTP(61)=1      ! Parton showering on or off'),
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring('pythiaUESettings', 
            'processParameters')
    )
)

process.GEN = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('TestTTbar_redo.root')
)

process.p = cms.Path(process.generator)
## process.p1 = cms.Path(process.randomEngineStateProducer)
process.outpath = cms.EndPath(process.GEN)

## process.schedule = cms.Schedule(process.p, process.p1, process.outpath)
process.schedule = cms.Schedule(process.p, process.outpath)
