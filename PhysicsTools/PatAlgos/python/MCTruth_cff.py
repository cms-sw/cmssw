import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.electronMatch_cfi import *
from PhysicsTools.PatAlgos.muonMatch_cfi import *
from PhysicsTools.PatAlgos.tauMatch_cfi import *
from PhysicsTools.PatAlgos.photonMatch_cfi import *
from PhysicsTools.PatAlgos.jetMatch_cfi import *
patMCTruth_withoutTau = cms.Sequence(electronMatch*muonMatch*photonMatch*jetPartonMatch*jetGenJetMatch)
patMCTruth = cms.Sequence(patMCTruth_withoutTau*tauMatch)

