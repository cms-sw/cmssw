import FWCore.ParameterSet.Config as cms

from ..sequences.HLTTiclLayerTileSequence_cfi import *
from ..sequences.HLTTiclPFSequence_cfi import *
from ..sequences.HLTTiclTrackstersCLUE3DHighStepSequence_cfi import *
from ..sequences.HLTTiclTrackstersRecoverySequence_cfi import *
from ..sequences.HLTTiclTracksterLinksSequence_cfi import *
from ..sequences.HLTTiclCandidateSequence_cfi import *
from ..sequences.HLTTiclTrackstersCLUE3DBarrelStepSequence_cfi import *

HLTIterTICLSequence = cms.Sequence(HLTTiclLayerTileSequence+HLTTiclTrackstersCLUE3DHighStepSequence+HLTTiclTrackstersRecoverySequence+HLTTiclTracksterLinksSequence+HLTTiclCandidateSequence+HLTTiclPFSequence)

from Configuration.ProcessModifiers.ticl_barrel_cff import ticl_barrel

_HLTIterTICLSequence_ticl_barrel = HLTIterTICLSequence.copy()
_HLTIterTICLSequence_ticl_barrel += HLTTiclTrackstersCLUE3DBarrelStepSequence

(ticl_barrel).toReplaceWith(HLTIterTICLSequence, _HLTIterTICLSequence_ticl_barrel)
