import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.mcMatchLayer0.electronMatch_cfi import *
from PhysicsTools.PatAlgos.mcMatchLayer0.muonMatch_cfi import *
from PhysicsTools.PatAlgos.mcMatchLayer0.tauMatch_cfi import *
from PhysicsTools.PatAlgos.mcMatchLayer0.photonMatch_cfi import *
from PhysicsTools.PatAlgos.mcMatchLayer0.jetMatch_cfi import *
patMCTruth_withoutTau = cms.Sequence(electronMatch*muonMatch*photonMatch*jetPartonMatch*jetGenJetMatch)
patMCTruth = cms.Sequence(patMCTruth_withoutTau*tauMatch)

