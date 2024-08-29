import FWCore.ParameterSet.Config as cms

from ..sequences.HLTTiclLayerTileSequence_cfi import *
from ..sequences.HLTTiclPFSequence_cfi import *
from ..sequences.HLTTiclTracksterMergeSequence_cfi import *
from ..sequences.HLTTiclTrackstersCLUE3DHighStepSequence_cfi import *
from ..sequences.HLTTiclTrackstersRecoverySequence_cfi import *
from ..sequences.HLTTiclTracksterLinksSequence_cfi import *
from ..sequences.HLTTiclCandidateSequence_cfi import *

HLTIterTICLSequence = cms.Sequence(HLTTiclLayerTileSequence+HLTTiclTrackstersCLUE3DHighStepSequence+HLTTiclTracksterMergeSequence+HLTTiclPFSequence)

from Configuration.ProcessModifiers.ticl_v5_cff import ticl_v5
ticl_v5.toReplaceWith(HLTIterTICLSequence, cms.Sequence(HLTTiclLayerTileSequence+HLTTiclTrackstersCLUE3DHighStepSequence+HLTTiclTrackstersRecoverySequence+HLTTiclTracksterLinksSequence+HLTTiclCandidateSequence+HLTTiclPFSequence))
