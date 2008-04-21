import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.selectionLayer1.electronMinFilter_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.electronMaxFilter_cfi import *
countLayer1Electrons = cms.Sequence(minLayer1Electrons+maxLayer1Electrons)

