import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.GeneratorMix_cff import *

pgen = cms.Sequence(cms.SequencePlaceholder("mix")+cms.SequencePlaceholder("generator")+cms.SequencePlaceholder("randomEngineStateProducer")+VertexSmearing+GenSmeared+genParticles+hiGenJets)
