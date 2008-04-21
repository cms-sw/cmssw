import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.selectionLayer1.photonMinFilter_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.photonMaxFilter_cfi import *
countLayer1Photons = cms.Sequence(minLayer1Photons+maxLayer1Photons)

