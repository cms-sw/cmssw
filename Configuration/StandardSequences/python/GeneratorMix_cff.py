import FWCore.ParameterSet.Config as cms

# Produce GenParticles of the two HepMCProducts
from Configuration.StandardSequences.Generator_cff import *
from GeneratorInterface.Core.generatorSmeared_cfi import *
genParticles.mix = cms.string("mix")
genParticles.doSubEvent = cms.untracked.bool(True)
genParticles.useCrossingFrame = cms.untracked.bool(True)
genParticles.saveBarCodes = cms.untracked.bool(True)
genParticles.abortOnUnknownPDGCode = cms.untracked.bool(False)

GenSmeared = cms.Sequence(generatorSmeared)
hiGenJets = cms.Sequence(genParticlesForJets*hiRecoGenJets)
pgen = cms.Sequence(cms.SequencePlaceholder("randomEngineStateProducer")+cms.SequencePlaceholder("mix")+VertexSmearing+GenSmeared+genParticles+hiGenJets)

