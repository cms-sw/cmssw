import FWCore.ParameterSet.Config as cms

from ..modules.siPixelClusters_cfi import *
from ..modules.siPixelRecHits_cfi import *

HLTDoLocalPixelTask = cms.Task(siPixelClusters, siPixelRecHits)
