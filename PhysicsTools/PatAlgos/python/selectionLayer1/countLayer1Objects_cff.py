import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.selectionLayer1.electronCountFilter_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.muonCountFilter_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.tauCountFilter_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.photonCountFilter_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.jetCountFilter_cfi import *

from PhysicsTools.PatAlgos.selectionLayer1.leptonCountFilter_cfi import *

countLayer1Objects = cms.Sequence(
    countLayer1Electrons +
    countLayer1Muons +
    countLayer1Taus +
    countLayer1Leptons +
    countLayer1Photons +
    countLayer1Jets 
)
