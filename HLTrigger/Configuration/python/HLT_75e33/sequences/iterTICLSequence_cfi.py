import FWCore.ParameterSet.Config as cms

from ..sequences.ticlLayerTileSequence_cfi import *
from ..sequences.ticlPFSequence_cfi import *
from ..sequences.ticlTracksterMergeSequence_cfi import *
from ..sequences.ticlTrackstersCLUE3DHighStepSequence_cfi import *

iterTICLSequence = cms.Sequence(ticlLayerTileSequence+ticlTrackstersCLUE3DHighStepSequence+ticlTracksterMergeSequence+ticlPFSequence)
