import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.selectionLayer1.electronCountFilter_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.muonCountFilter_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.tauCountFilter_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.photonCountFilter_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.jetCountFilter_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.leptonCountFilter_cfi import *

countPatCandidates = cms.Sequence(
    countPatElectrons +
    countPatMuons     +
    countPatTaus      +
    countPatLeptons   +
    countPatPhotons   +
    countPatJets 
)
