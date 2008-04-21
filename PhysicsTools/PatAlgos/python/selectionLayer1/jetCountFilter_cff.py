import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.selectionLayer1.jetMinFilter_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.jetMaxFilter_cfi import *
countLayer1Jets = cms.Sequence(minLayer1Jets+maxLayer1Jets)

