import FWCore.ParameterSet.Config as cms

from ..modules.hltFixedGridRhoFastjetAllCaloForMuons_cfi import *

HLTFastJetForEgammaTask = cms.Task(
    hltFixedGridRhoFastjetAllCaloForMuons
)
