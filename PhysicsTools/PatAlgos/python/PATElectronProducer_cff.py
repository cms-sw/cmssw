import FWCore.ParameterSet.Config as cms

from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *
from MagneticField.Engine.volumeBasedMagneticField_cfi import *
from PhysicsTools.PatAlgos.PATElectronProducer_cfi import *
from PhysicsTools.PatAlgos.PATElectronSelector_cfi import *
from PhysicsTools.PatAlgos.PATElectronCountFilter_cff import *
layer1Electrons = cms.Sequence(allLayer1Electrons*selectedLayer1Electrons*countLayer1Electrons)

