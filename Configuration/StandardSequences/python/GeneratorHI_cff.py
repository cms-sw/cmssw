import FWCore.ParameterSet.Config as cms

# Produce GenParticles of the two HepMCProducts
from Configuration.StandardSequences.Generator_cff import *

genParticles.doSubEvent = cms.untracked.bool(True)

hiGenJets = cms.Sequence(genParticlesForJets*hiRecoGenJets)
pgen = cms.Sequence(cms.SequencePlaceholder("randomEngineStateProducer")+VertexSmearing+genParticles+hiGenJets)

