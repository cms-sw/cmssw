import FWCore.ParameterSet.Config as cms

from ..modules.hltEcalBarrelClusterFastTimer_cfi import *
from ..modules.hltParticleFlowClusterECAL_cfi import *
from ..modules.hltParticleFlowTimeAssignerECAL_cfi import *

HLTParticleFlowClusterECALSequence = cms.Sequence(hltEcalBarrelClusterFastTimer+hltParticleFlowTimeAssignerECAL+hltParticleFlowClusterECAL)
