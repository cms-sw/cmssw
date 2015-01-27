import FWCore.ParameterSet.Config as cms

# Produce GenParticles of the two HepMCProducts
from Configuration.StandardSequences.Generator_cff import *
genParticles.mix = cms.string("mix")
genParticles.doSubEvent = cms.untracked.bool(True)
genParticles.useCrossingFrame = cms.untracked.bool(True)
genParticles.saveBarCodes = cms.untracked.bool(True)
genParticles.abortOnUnknownPDGCode = cms.untracked.bool(False)

hiGenJets = cms.Sequence(genParticlesForJets*hiRecoGenJets)
pgen = cms.Sequence(cms.SequencePlaceholder("randomEngineStateProducer")+cms.SequencePlaceholder("mix")+VertexSmearing+genParticles+hiGenJets)

