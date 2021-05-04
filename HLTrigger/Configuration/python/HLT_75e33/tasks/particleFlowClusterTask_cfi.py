import FWCore.ParameterSet.Config as cms

from ..modules.particleFlowBadHcalPseudoCluster_cfi import *
from ..tasks.pfClusteringECALTask_cfi import *
from ..tasks.pfClusteringHBHEHFTask_cfi import *
from ..tasks.pfClusteringHGCalTask_cfi import *
from ..tasks.pfClusteringHOTask_cfi import *
from ..tasks.pfClusteringPSTask_cfi import *

particleFlowClusterTask = cms.Task(
    particleFlowBadHcalPseudoCluster,
    pfClusteringECALTask,
    pfClusteringHBHEHFTask,
    pfClusteringHGCalTask,
    pfClusteringHOTask,
    pfClusteringPSTask
)
