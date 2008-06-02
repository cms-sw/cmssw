import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.PATPhotonMinFilter_cfi import *
from PhysicsTools.PatAlgos.PATPhotonMaxFilter_cfi import *
countLayer1Photons = cms.Sequence(minLayer1Photons+maxLayer1Photons)

