import FWCore.ParameterSet.Config as cms

# make allLayer1Objects
from PhysicsTools.PatAlgos.producersLayer1.allLayer1Objects_cff import *

# make selectedLayer1Objects
from PhysicsTools.PatAlgos.selectionLayer1.selectedLayer1Objects_cff import *

# make cleanLayer1Objects
from PhysicsTools.PatAlgos.cleaningLayer1.cleanLayer1Objects_cff import *

# count selected layer 1 objects (including total number of leptons)
from PhysicsTools.PatAlgos.selectionLayer1.countLayer1Objects_cff import *

patDefaultSequence = cms.Sequence(
    allLayer1Objects * 
    selectedLayer1Objects *
    cleanLayer1Objects *
    countLayer1Objects
)

patDefaultSequenceNoCleaning = cms.Sequence(
    allLayer1Objects * 
    selectedLayer1Objects *
    countLayer1Objects
)
