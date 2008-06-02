import FWCore.ParameterSet.Config as cms

from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *
from MagneticField.Engine.volumeBasedMagneticField_cfi import *
from PhysicsTools.PatAlgos.PATMuonProducer_cfi import *
from PhysicsTools.PatAlgos.PATMuonSelector_cfi import *
from PhysicsTools.PatAlgos.PATMuonCountFilter_cff import *
layer1Muons = cms.Sequence(allLayer1Muons*selectedLayer1Muons*countLayer1Muons)

