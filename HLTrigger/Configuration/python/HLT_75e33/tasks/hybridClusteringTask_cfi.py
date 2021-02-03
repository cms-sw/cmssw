import FWCore.ParameterSet.Config as cms

from ..modules.cleanedHybridSuperClusters_cfi import *
from ..modules.correctedHybridSuperClusters_cfi import *
from ..modules.hybridSuperClusters_cfi import *
from ..modules.uncleanedHybridSuperClusters_cfi import *
from ..modules.uncleanedOnlyCorrectedHybridSuperClusters_cfi import *

hybridClusteringTask = cms.Task(cleanedHybridSuperClusters, correctedHybridSuperClusters, hybridSuperClusters, uncleanedHybridSuperClusters, uncleanedOnlyCorrectedHybridSuperClusters)
