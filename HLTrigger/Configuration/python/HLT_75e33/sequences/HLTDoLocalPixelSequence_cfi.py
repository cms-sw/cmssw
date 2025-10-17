import FWCore.ParameterSet.Config as cms

from ..modules.hltSiPixelClusters_cfi import *
from ..modules.hltSiPixelRecHits_cfi import *
from ..modules.hltPhase2SiPixelClustersSoA_cfi import hltPhase2SiPixelClustersSoA
from ..modules.hltPhase2SiPixelRecHitsSoA_cfi  import hltPhase2SiPixelRecHitsSoA
from ..modules.hltSiPixelClusterShapeCache_cfi import hltSiPixelClusterShapeCache

HLTDoLocalPixelSequence = cms.Sequence(
     hltPhase2SiPixelClustersSoA
    +hltSiPixelClusters
    +hltSiPixelClusterShapeCache  # should we disable this? Still needed by tracker muons
    +hltPhase2SiPixelRecHitsSoA
    +hltSiPixelRecHits
)
