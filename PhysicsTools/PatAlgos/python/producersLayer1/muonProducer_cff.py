import FWCore.ParameterSet.Config as cms

from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *
from MagneticField.Engine.volumeBasedMagneticField_cfi import *
from PhysicsTools.PatAlgos.producersLayer1.muonProducer_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.muonSelector_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.muonCountFilter_cff import *
layer1Muons = cms.Sequence(allLayer1Muons*selectedLayer1Muons*countLayer1Muons)

