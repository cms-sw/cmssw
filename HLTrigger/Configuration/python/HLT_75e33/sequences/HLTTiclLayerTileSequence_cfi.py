import FWCore.ParameterSet.Config as cms

from ..modules.hltTiclLayerTileProducer_cfi import *

HLTTiclLayerTileSequence = cms.Sequence(hltTiclLayerTileProducer)
