import FWCore.ParameterSet.Config as cms

from MagneticField.Engine.volumeBasedMagneticField_cfi import *
from PhysicsTools.PatAlgos.PATTauProducer_cfi import *
from PhysicsTools.PatAlgos.PATTauSelector_cfi import *
from PhysicsTools.PatAlgos.PATTauCountFilter_cff import *
layer1Taus = cms.Sequence(allLayer1Taus*selectedLayer1Taus*countLayer1Taus)

