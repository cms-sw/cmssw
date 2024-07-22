import FWCore.ParameterSet.Config as cms

from ..sequences.ticlLayerTileSequence_cfi import *
from ..sequences.ticlPFSequence_cfi import *
from ..sequences.ticlTracksterMergeSequence_cfi import *
from ..sequences.ticlTrackstersCLUE3DHighStepSequence_cfi import *
from ..sequences.HLTTiclTrackstersPassthroughSequence_cfi import *
from ..sequences.HLTTiclTracksterLinksSequence_cfi import *
from ..sequences.HLTTiclCandidateSequence_cfi import *

iterTICLSequence = cms.Sequence(ticlLayerTileSequence+ticlTrackstersCLUE3DHighStepSequence+ticlTracksterMergeSequence+ticlPFSequence)

from Configuration.ProcessModifiers.ticl_v5_cff import ticl_v5
ticl_v5.toReplaceWith(iterTICLSequence, cms.Sequence(ticlLayerTileSequence+ticlTrackstersCLUE3DHighStepSequence+HLTTiclTrackstersPassthroughSequence+HLTTiclTracksterLinksSequence+HLTTiclCandidateSequence+ticlPFSequence))
