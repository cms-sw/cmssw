import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.PATTauMinFilter_cfi import *
from PhysicsTools.PatAlgos.PATTauMaxFilter_cfi import *
countLayer1Taus = cms.Sequence(minLayer1Taus+maxLayer1Taus)

