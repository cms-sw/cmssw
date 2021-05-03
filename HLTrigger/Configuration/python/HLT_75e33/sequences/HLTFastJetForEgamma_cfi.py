import FWCore.ParameterSet.Config as cms

from ..tasks.HLTFastJetForEgammaTask_cfi import *

HLTFastJetForEgamma = cms.Sequence(HLTFastJetForEgammaTask)
