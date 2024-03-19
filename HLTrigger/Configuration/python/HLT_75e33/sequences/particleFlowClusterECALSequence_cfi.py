import FWCore.ParameterSet.Config as cms

from ..modules.ecalBarrelClusterFastTimer_cfi import *
from ..modules.particleFlowClusterECAL_cfi import *
from ..modules.particleFlowTimeAssignerECAL_cfi import *

particleFlowClusterECALSequence = cms.Sequence(ecalBarrelClusterFastTimer+particleFlowTimeAssignerECAL+particleFlowClusterECAL)
