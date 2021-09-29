import FWCore.ParameterSet.Config as cms

from ..modules.hltTowerMakerForAllForEgamma_cfi import *

HLTFastJetForEgammaTask = cms.Task(
    hltTowerMakerForAllForEgamma
)
