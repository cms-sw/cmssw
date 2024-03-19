import FWCore.ParameterSet.Config as cms

from ..modules.ticlLayerTileProducer_cfi import *

ticlLayerTileSequence = cms.Sequence(ticlLayerTileProducer)
