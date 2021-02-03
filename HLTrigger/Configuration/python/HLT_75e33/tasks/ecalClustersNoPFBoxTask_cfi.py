import FWCore.ParameterSet.Config as cms

from ..tasks.hybridClusteringTask_cfi import *
from ..tasks.multi5x5ClusteringTask_cfi import *
from ..tasks.multi5x5PreshowerClusteringTask_cfi import *

ecalClustersNoPFBoxTask = cms.Task(hybridClusteringTask, multi5x5ClusteringTask, multi5x5PreshowerClusteringTask)
