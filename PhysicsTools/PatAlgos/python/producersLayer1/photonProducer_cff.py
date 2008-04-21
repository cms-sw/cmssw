import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.producersLayer1.photonProducer_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.photonSelector_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.photonCountFilter_cff import *
layer1Photons = cms.Sequence(allLayer1Photons*selectedLayer1Photons*countLayer1Photons)

