import FWCore.ParameterSet.Config as cms

# Produce GenParticles of the two HepMCProducts
from Configuration.StandardSequences.Generator_cff import *
from GeneratorInterface.Core.generatorSmeared_cfi import *

genParticles.doSubEvent = cms.untracked.bool(True)

GenSmeared = cms.Sequence(generatorSmeared)
hiGenJets = cms.Sequence(genParticlesForJets*hiRecoGenJets)
pgen = cms.Sequence(cms.SequencePlaceholder("randomEngineStateProducer")+VertexSmearing+GenSmeared+genParticles+hiGenJets)

