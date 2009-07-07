#!/usr/bin/env cmsRun
import FWCore.ParameterSet.Config as cms

process = cms.Process("GEN")

process.source = cms.Source("AlpgenSource",
        fileNames = cms.untracked.vstring('file:w2j') 
)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(100))

process.load("Configuration.StandardSequences.Services_cff")

# Setup RandomNumberGeneratorService so it provides
# random numbers to the Alpgen Producer.
process.RandomNumberGeneratorService.generator = cms.PSet(
        initialSeed = cms.untracked.uint32(123456789),
        engineName = cms.untracked.string('HepJamesRandom')
)

# The Alpgen Producer.
# Change the jet matching parameters as you see fit.
process.load("GeneratorInterface.AlpgenInterface.generator_cfi")
process.generator.maxEventsToPrint = 0
process.generator.jetMatching.applyMatching = True
process.generator.jetMatching.exclusive = True
process.generator.jetMatching.etMin = 25.0
process.generator.jetMatching.drMin = 0.7

process.p0 = cms.Path(process.generator)

# Alternatively, you may also want to use these,
# if you want a more complete generation.

#process.load("Configuration.StandardSequences.Generator_cff")
#process.load("Configuration.StandardSequences.VtxSmearedGauss_cff")

#process.VtxSmeared.src = 'generator'
#process.genParticles.src = 'generator'
#process.genParticleCandidates.src = 'generator'

# Needed for the SIM step.
#process.g4SimHits.Generator.HepMCProductLabel = 'generator'
#process.mergedtruth.HepMCDataLabels.append('generator')

# Comment the path above if you want to use this one.
#process.p0 = cms.Path(process.generator * process.pgen)

process.load("Configuration.EventContent.EventContent_cff")

process.GEN = cms.OutputModule("PoolOutputModule",
	process.FEVTSIMEventContent,
	dataset = cms.untracked.PSet(dataTier = cms.untracked.string('GEN')),
	SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('p0')),
	fileName = cms.untracked.string('test.root')
)
process.GEN.outputCommands.append("keep *_generator_*_*")

process.outpath = cms.EndPath(process.GEN)
