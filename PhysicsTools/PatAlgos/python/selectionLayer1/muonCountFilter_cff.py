import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.selectionLayer1.muonMinFilter_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.muonMaxFilter_cfi import *
countLayer1Muons = cms.Sequence(minLayer1Muons+maxLayer1Muons)

