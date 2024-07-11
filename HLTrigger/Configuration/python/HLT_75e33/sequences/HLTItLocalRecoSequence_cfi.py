import FWCore.ParameterSet.Config as cms

from ..modules.hltSiPhase2Clusters_cfi import *
from ..modules.hltSiPixelClusters_cfi import *
from ..modules.hltSiPixelClusterShapeCache_cfi import *
from ..modules.hltSiPixelRecHits_cfi import *

HLTItLocalRecoSequence = cms.Sequence(hltSiPhase2Clusters+hltSiPixelClusters+hltSiPixelClusterShapeCache+hltSiPixelRecHits)
