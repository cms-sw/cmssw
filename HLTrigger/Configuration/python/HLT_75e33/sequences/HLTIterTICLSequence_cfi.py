import FWCore.ParameterSet.Config as cms

from ..sequences.HLTTiclLayerTileSequence_cfi import *
from ..sequences.HLTTiclPFSequence_cfi import *
from ..sequences.HLTTiclTracksterMergeSequence_cfi import *
from ..sequences.HLTTiclTrackstersCLUE3DHighStepSequence_cfi import *
from ..sequences.HLTTiclTrackstersRecoverySequence_cfi import *
from ..sequences.HLTTiclTracksterLinksSequence_cfi import *
from ..sequences.HLTTiclCandidateSequence_cfi import *
from ..sequences.HLTTiclTrackstersCLUE3DBarrelStepSequence_cfi import *

HLTIterTICLSequence = cms.Sequence(HLTTiclLayerTileSequence+HLTTiclTrackstersCLUE3DHighStepSequence+HLTTiclTracksterMergeSequence+HLTTiclPFSequence)

_HLTIterTICLSequence_ticl_v5 = cms.Sequence(
                                  HLTTiclLayerTileSequence+
                                  HLTTiclTrackstersCLUE3DHighStepSequence+
                                  HLTTiclTrackstersRecoverySequence+
                                  HLTTiclTracksterLinksSequence+
                                  HLTTiclCandidateSequence+
                                  HLTTiclPFSequence
)

from Configuration.ProcessModifiers.ticl_v5_cff import ticl_v5
from Configuration.ProcessModifiers.ticl_barrel_cff import ticl_barrel

(ticl_v5 & (~ticl_barrel)).toReplaceWith(HLTIterTICLSequence, _HLTIterTICLSequence_ticl_v5)

_HLTIterTICLSequence_ticl_barrel = HLTIterTICLSequence.copy()
_HLTIterTICLSequence_ticl_barrel += HLTTiclTrackstersCLUE3DBarrelStepSequence

_HLTIterTICLSequence_ticl_v5_barrel = _HLTIterTICLSequence_ticl_v5.copy()
_HLTIterTICLSequence_ticl_v5_barrel += HLTTiclTrackstersCLUE3DBarrelStepSequence

(ticl_barrel & (~ticl_v5)).toReplaceWith(HLTIterTICLSequence, _HLTIterTICLSequence_ticl_barrel)
(ticl_barrel & ticl_v5).toReplaceWith(HLTIterTICLSequence, _HLTIterTICLSequence_ticl_v5_barrel)
