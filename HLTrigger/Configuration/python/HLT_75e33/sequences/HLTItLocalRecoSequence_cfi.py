import FWCore.ParameterSet.Config as cms

from ..modules.hltSiPhase2Clusters_cfi import *
from ..modules.hltSiPixelClusters_cfi import *
from ..modules.hltSiPixelClusterShapeCache_cfi import *
from ..modules.hltSiPixelRecHits_cfi import *

HLTItLocalRecoSequence = cms.Sequence(hltSiPhase2Clusters+hltSiPixelClusters+hltSiPixelClusterShapeCache+hltSiPixelRecHits)

from ..sequences.HLTDoLocalPixelSequence_cfi import *
from ..sequences.HLTDoLocalStripSequence_cfi import *
_HLTItLocalRecoSequence = cms.Sequence(HLTDoLocalPixelSequence+HLTDoLocalStripSequence)

from Configuration.ProcessModifiers.alpaka_cff import alpaka
alpaka.toReplaceWith(HLTItLocalRecoSequence, _HLTItLocalRecoSequence)
