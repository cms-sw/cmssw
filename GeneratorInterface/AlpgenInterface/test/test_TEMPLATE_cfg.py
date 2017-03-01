#!/usr/bin/env cmsRun
import FWCore.ParameterSet.Config as cms

# The values in the r.h.s are supposed to be changed in an automatic
# way, e.g. sed 's/MATCHING/True/'
AlpgenCard = "FILENAME"
AlpgenApplyMatching = MATCHING
AlpgenExclusive = EXCLUSIVE
AlpgenEtMin = ETMIN
AlpgenDrMin = DRMIN

process = cms.Process("GEN")

process.source = cms.Source("AlpgenSource",
        fileNames = cms.untracked.vstring('file:'+AlpgenCard) 
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
process.generator.jetMatching.applyMatching = AlpgenApplyMatching
process.generator.jetMatching.exclusive = AlpgenExclusive
process.generator.jetMatching.etMin = AlpgenEtMin
process.generator.jetMatching.drMin = AlpgenDrMin

process.p0 = cms.Path(process.generator)

# Alternatively, you may also want to use these,
# if you want a more complete generation.

#process.load("Configuration.StandardSequences.Generator_cff")
#process.load("Configuration.StandardSequences.VtxSmearedGauss_cff")

#process.VtxSmeared.src = 'generator'
#process.genParticles.src = 'VtxSmeared'
#process.genParticleCandidates.src = 'VtxSmeared'

# Needed for the SIM step.
#process.g4SimHits.Generator.HepMCProductLabel = 'VtxSmeared'
#process.mergedtruth.HepMCDataLabels.append('VtxSmeared')

# Comment the path above if you want to use this one.
#process.p0 = cms.Path(process.generator * process.pgen)

process.load("Configuration.EventContent.EventContent_cff")

process.GEN = cms.OutputModule("PoolOutputModule",
	process.FEVTSIMEventContent,
	dataset = cms.untracked.PSet(dataTier = cms.untracked.string('GEN')),
	SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('p0')),
	fileName = cms.untracked.string(AlpgenCard+'.root')
)
process.GEN.outputCommands.append("keep *_generator_*_*")

process.outpath = cms.EndPath(process.GEN)
