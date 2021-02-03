import FWCore.ParameterSet.Config as cms

from ..modules.siPhase2Clusters_cfi import *
from ..modules.siPixelClusters_cfi import *
from ..modules.siPixelClusterShapeCache_cfi import *
from ..modules.siPixelRecHits_cfi import *

itLocalReco = cms.Sequence(siPhase2Clusters+siPixelClusters+siPixelClusterShapeCache+siPixelRecHits)
