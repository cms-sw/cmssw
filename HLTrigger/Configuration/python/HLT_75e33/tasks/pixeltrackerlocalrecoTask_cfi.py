import FWCore.ParameterSet.Config as cms

from ..modules.siPhase2Clusters_cfi import *
from ..modules.siPixelClustersPreSplitting_cfi import *
from ..modules.siPixelRecHitsPreSplitting_cfi import *

pixeltrackerlocalrecoTask = cms.Task(siPhase2Clusters, siPixelClustersPreSplitting, siPixelRecHitsPreSplitting)
