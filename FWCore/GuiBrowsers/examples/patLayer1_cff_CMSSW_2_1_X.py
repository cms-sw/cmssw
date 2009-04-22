import FWCore.ParameterSet.Config as cms

#  Layer 1 default sequence
# build the Objects (Jets, Muons, Electrons, METs, Taus)
from PhysicsTools.PatAlgos.producersLayer1.electronProducer_cff import *
from PhysicsTools.PatAlgos.producersLayer1.muonProducer_cff import *
from PhysicsTools.PatAlgos.producersLayer1.tauProducer_cff import *
from PhysicsTools.PatAlgos.producersLayer1.photonProducer_cff import *
from PhysicsTools.PatAlgos.producersLayer1.jetProducer_cff import *
from PhysicsTools.PatAlgos.producersLayer1.metProducer_cff import *
from PhysicsTools.PatAlgos.producersLayer1.hemisphereProducer_cff import *
#FIXME: Why do we need this here?
from PhysicsTools.PatAlgos.selectionLayer1.leptonCountFilter_cfi import *
allObjects = cms.Sequence(layer1Muons*layer1Electrons*layer1Taus*countLayer1Leptons*layer1Photons*layer1Jets*layer1METs*layer1Hemispheres)
patLayer1 = cms.Sequence(allObjects)

