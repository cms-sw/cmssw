import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.recoLayer0.electronIsolation_cff import *
from PhysicsTools.PatAlgos.recoLayer0.muonIsolation_cff import *
from PhysicsTools.PatAlgos.recoLayer0.photonIsolation_cff import *
from PhysicsTools.PatAlgos.recoLayer0.jetFlavourId_cff import *
from PhysicsTools.PatAlgos.recoLayer0.bTagging_cff import *
patHighLevelReco_withoutPFTau = cms.Sequence(patLayer0ElectronIsolation*patLayer0PhotonIsolation*patLayer0MuonIsolation)
patHighLevelReco = cms.Sequence(patHighLevelReco_withoutPFTau)

