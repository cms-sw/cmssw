import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.PATJetMinFilter_cfi import *
from PhysicsTools.PatAlgos.PATJetMaxFilter_cfi import *
countLayer1Jets = cms.Sequence(minLayer1Jets+maxLayer1Jets)

