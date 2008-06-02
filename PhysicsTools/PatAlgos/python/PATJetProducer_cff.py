import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.PATJetProducer_cfi import *
from PhysicsTools.PatAlgos.PATJetSelector_cfi import *
from PhysicsTools.PatAlgos.PATJetCountFilter_cff import *
layer1Jets = cms.Sequence(allLayer1Jets*selectedLayer1Jets*countLayer1Jets)

