import FWCore.ParameterSet.Config as cms

from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *
from MagneticField.Engine.volumeBasedMagneticField_cfi import *
from PhysicsTools.PatAlgos.producersLayer1.electronProducer_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.electronSelector_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.electronCountFilter_cff import *
layer1Electrons = cms.Sequence(allLayer1Electrons*selectedLayer1Electrons*countLayer1Electrons)

