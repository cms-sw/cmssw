import FWCore.ParameterSet.Config as cms

from ..modules.particleFlowSuperClusterECAL_cfi import *

particleFlowSuperClusteringSequence = cms.Sequence(particleFlowSuperClusterECAL)
