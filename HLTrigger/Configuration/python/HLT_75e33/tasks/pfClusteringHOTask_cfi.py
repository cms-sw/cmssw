import FWCore.ParameterSet.Config as cms

from ..modules.particleFlowClusterHO_cfi import *
from ..modules.particleFlowRecHitHO_cfi import *

pfClusteringHOTask = cms.Task(
    particleFlowClusterHO,
    particleFlowRecHitHO
)
# foo bar baz
# ugxAJVFrVsVq2
# tBoINPUXhXYEg
