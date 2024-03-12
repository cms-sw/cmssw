import FWCore.ParameterSet.Config as cms

from ..modules.siPixelClusters_cfi import *
from ..modules.siPixelRecHits_cfi import *

HLTDoLocalPixelSequence = cms.Sequence(siPixelClusters+siPixelRecHits)
