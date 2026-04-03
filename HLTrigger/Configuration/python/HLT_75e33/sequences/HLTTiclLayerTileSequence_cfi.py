import FWCore.ParameterSet.Config as cms

from ..modules.hltTiclLayerTileProducer_cfi import *
from ..modules.hltTiclLayerTileBarrelProducer_cfi import *

HLTTiclLayerTileSequence = cms.Sequence(hltTiclLayerTileProducer)

from Configuration.ProcessModifiers.ticl_barrel_cff import ticl_barrel
ticl_barrel.toReplaceWith(HLTTiclLayerTileSequence, cms.Sequence(hltTiclLayerTileProducer+hltTiclLayerTileBarrelProducer))
