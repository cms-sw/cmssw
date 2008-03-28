import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.PATElectronProducer_cff import *
from PhysicsTools.PatAlgos.PATMuonProducer_cff import *
from PhysicsTools.PatAlgos.PATTauProducer_cff import *
from PhysicsTools.PatAlgos.PATPhotonProducer_cff import *
from PhysicsTools.PatAlgos.PATJetProducer_cff import *
from PhysicsTools.PatAlgos.PATMETProducer_cff import *
from PhysicsTools.PatAlgos.PATLeptonCountFilter_cfi import *
allObjects = cms.Sequence(layer1Muons*layer1Electrons*layer1Taus*countLayer1Leptons*layer1Photons*layer1Jets*layer1METs)

