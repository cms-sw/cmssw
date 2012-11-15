import FWCore.ParameterSet.Config as cms

# make patCandidates
from PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff import *

# make selectedPatCandidates
from PhysicsTools.PatAlgos.selectionLayer1.selectedPatCandidates_cff import *

# make cleanPatCandidates
from PhysicsTools.PatAlgos.cleaningLayer1.cleanPatCandidates_cff import *

# count cleanPatCandidates (including total number of leptons)
from PhysicsTools.PatAlgos.selectionLayer1.countPatCandidates_cff import *

patDefaultSequence = cms.Sequence(
# remove this (particleFlowPtrs) after we switch to unscheduled mode everywhere
# too many places to change otherwise
    particleFlowPtrs *
    patCandidates * 
    selectedPatCandidates *
    cleanPatCandidates *
    countPatCandidates
)
