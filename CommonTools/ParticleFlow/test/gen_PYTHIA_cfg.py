

# event generation
# analysis of the gen event

import FWCore.ParameterSet.Config as cms


process = cms.Process("GEN")


# event generation ------------------------------------------------------



process.load("FastSimulation/Configuration/RandomServiceInitialization_cff")
process.load("CommonTools.ParticleFlow.Sources/source_ZtoJets_cfi")
process.RandomNumberGeneratorService.theSource.initialSeed = 13124

# path  -----------------------------------------------------------------

process.source.maxEventsToPrint = cms.untracked.int32(1)
process.source.pythiaPylistVerbosity = cms.untracked.int32(1)
process.source.pythiaHepMCVerbosity = cms.untracked.bool(False)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)


#process.p1 = cms.Path(
#    process.genParticles
#    )


