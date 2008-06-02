import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.PATElectronMinFilter_cfi import *
from PhysicsTools.PatAlgos.PATElectronMaxFilter_cfi import *
countLayer1Electrons = cms.Sequence(minLayer1Electrons+maxLayer1Electrons)

