import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.selectionLayer1.tauMinFilter_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.tauMaxFilter_cfi import *
countLayer1Taus = cms.Sequence(minLayer1Taus+maxLayer1Taus)

