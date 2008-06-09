import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.producersLayer1.jetProducer_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.jetSelector_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.jetCountFilter_cff import *
layer1Jets = cms.Sequence(allLayer1Jets*selectedLayer1Jets*countLayer1Jets)

