import FWCore.ParameterSet.Config as cms

from ..modules.hltSiPixelClusters_cfi import *
from ..modules.hltSiPixelRecHits_cfi import *

HLTDoLocalPixelSequence = cms.Sequence(hltSiPixelClusters+hltSiPixelRecHits)
