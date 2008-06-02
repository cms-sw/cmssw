import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.PATMuonMinFilter_cfi import *
from PhysicsTools.PatAlgos.PATMuonMaxFilter_cfi import *
countLayer1Muons = cms.Sequence(minLayer1Muons+maxLayer1Muons)

