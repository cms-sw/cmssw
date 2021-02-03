import FWCore.ParameterSet.Config as cms

from ..modules.multi5x5BasicClustersCleaned_cfi import *
from ..modules.multi5x5BasicClustersUncleaned_cfi import *
from ..modules.multi5x5SuperClusters_cfi import *
from ..modules.multi5x5SuperClustersCleaned_cfi import *
from ..modules.multi5x5SuperClustersUncleaned_cfi import *
from ..modules.multi5x5SuperClustersWithPreshower_cfi import *

multi5x5ClusteringTask = cms.Task(multi5x5BasicClustersCleaned, multi5x5BasicClustersUncleaned, multi5x5SuperClusters, multi5x5SuperClustersCleaned, multi5x5SuperClustersUncleaned, multi5x5SuperClustersWithPreshower)
