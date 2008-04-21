import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.producersLayer1.metProducer_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.metSelector_cfi import *
layer1METs = cms.Sequence(allLayer1METs*selectedLayer1METs)

