import FWCore.ParameterSet.Config as cms

# make allLayer1Objects
from PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff import *

# make selectedLayer1Objects
from PhysicsTools.PatAlgos.selectionLayer1.selectedPatCandidates_cff import *

# make cleanLayer1Objects
from PhysicsTools.PatAlgos.cleaningLayer1.cleanPatCandidates_cff import *

# count selected layer 1 objects (including total number of leptons)
from PhysicsTools.PatAlgos.selectionLayer1.countPatCandidates_cff import *

patDefaultSequence = cms.Sequence(
    patCandidates * 
    selectedPatCandidates *
    cleanPatCandidates *
    countPatCandidates
)
