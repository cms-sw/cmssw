#!/usr/bin/env cmsRun
import FWCore.ParameterSet.Config as cms

process = cms.Process("GEN")

# POOL Source containing the output from a previous Alpgen Source.  
process.source = cms.Source("PoolSource",
        fileNames = cms.untracked.vstring('file:testSource.root') 
)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(100))

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.Services_cff")

# The Pythia6-based "hadronizer"
# Change the jet matching parameters as you see fit.
process.load("GeneratorInterface.AlpgenInterface.generator_cfi")
process.generator.maxEventsToPrint = 0
process.generator.jetMatching.applyMatching = True
process.generator.jetMatching.exclusive = True
process.generator.jetMatching.etMin = 25.0
process.generator.jetMatching.drMin = 0.7

process.p0 = cms.Path(process.generator)

process.load("Configuration.EventContent.EventContent_cff")

process.GEN = cms.OutputModule("PoolOutputModule",
	process.FEVTSIMEventContent,
	dataset = cms.untracked.PSet(dataTier = cms.untracked.string('GEN')),
	SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('p0')),
	fileName = cms.untracked.string('test.root')
)

process.outpath = cms.EndPath(process.GEN)
