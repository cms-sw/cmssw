import FWCore.ParameterSet.Config as cms

from FastSimulation.HighLevelTrigger.DummyModule_cfi import *
pixeltrackerlocalreco = cms.Sequence(dummyModule)
striptrackerlocalreco = cms.Sequence(dummyModule)
trackerlocalreco = cms.Sequence(dummyModule)

