import FWCore.ParameterSet.Config as cms

from ..tasks.ecalClustersTask_cfi import *

ecalClusters = cms.Sequence(ecalClustersTask)
