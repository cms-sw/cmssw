import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.PATPhotonProducer_cfi import *
from PhysicsTools.PatAlgos.PATPhotonSelector_cfi import *
from PhysicsTools.PatAlgos.PATPhotonCountFilter_cff import *
layer1Photons = cms.Sequence(allLayer1Photons*selectedLayer1Photons*countLayer1Photons)

