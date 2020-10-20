#!/usr/bin/env cmsRun
import FWCore.ParameterSet.Config as cms

from Configuration.Generator.PythiaUESettings_cfi import *

process = cms.Process("TEST")


process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    generator = cms.PSet(
        initialSeed = cms.untracked.uint32(123456789),
        engineName = cms.untracked.string('HepJamesRandom')
    )
)

process.source = cms.Source("AlpgenSource",
	fileNames = cms.untracked.vstring('file:w2j')
)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1)) # put here -1, if you want the whole

process.configurationMetadata = cms.untracked.PSet(
	version = cms.untracked.string('alpha'),
	name = cms.untracked.string('w2j'),
	annotation = cms.untracked.string('AG UNW -> LHE translation, Py6 hadronization')
)

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.destinations = ['cerr']
process.MessageLogger.statistics = []


process.generator = cms.EDFilter("Pythia6HadronizerFilter",
    pythiaHepMCVerbosity = cms.untracked.bool(True),
    maxEventsToPrint = cms.untracked.int32(0),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    comEnergy = cms.double(10000.0), # this will be disregarded in case of processing ME partons
    PythiaParameters = cms.PSet(
        pythiaUESettingsBlock,
        processParameters = cms.vstring
	(
	   'MSEL=0         ! User defined processes', 
	   'MSTJ(1)=1      ! Fragmentation/hadronization on or off'
	   'MSTJ(11)=3     ! Choice of fragmentation function'
	),
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring
	(
	   'pythiaUESettings', 
           'processParameters'
	)
    ),
    jetMatching = cms.untracked.PSet(
        scheme = cms.string("Alpgen"),
        applyMatching = cms.bool(True),
        exclusive = cms.bool(True),
        etMin = cms.double(25.),
        drMin = cms.double(0.7)
    )

)

###process.load("Configuration.EventContent.EventContent_cff")

process.USER = cms.OutputModule("PoolOutputModule",
   ###process.FEVTSIMEventContent,
   fileName = cms.untracked.string('testAGPy6Had.root'),
   SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('p'))
)

###process.LHE.outputCommands.append("keep *_source_*_*")

process.p = cms.Path(process.generator)
process.outpath = cms.EndPath(process.USER)

process.schedule = cms.Schedule(process.p, process.outpath)

