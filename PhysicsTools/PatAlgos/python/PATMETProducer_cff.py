import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.PATMETProducer_cfi import *
from PhysicsTools.PatAlgos.PATMETSelector_cfi import *
layer1METs = cms.Sequence(allLayer1METs*selectedLayer1METs)

