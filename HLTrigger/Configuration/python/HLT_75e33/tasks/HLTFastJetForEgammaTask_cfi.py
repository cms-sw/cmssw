import FWCore.ParameterSet.Config as cms

from ..modules.hltFixedGridRhoFastjetAllCaloForEGamma_cfi import *

HLTFastJetForEgammaTask = cms.Task(
    hltFixedGridRhoFastjetAllCaloForEGamma
)
