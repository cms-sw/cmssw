import FWCore.ParameterSet.Config as cms

from ..modules.hltParticleFlowClusterECALUncorrectedUnseeded_cfi import *
from ..modules.hltParticleFlowClusterECALUnseeded_cfi import *
from ..modules.hltParticleFlowRecHitECALUnseeded_cfi import *
from ..modules.hltParticleFlowSuperClusterECALUnseeded_cfi import *

HLTPFClusteringForEgammaUnseededTask = cms.Task(
    hltParticleFlowClusterECALUncorrectedUnseeded,
    hltParticleFlowClusterECALUnseeded,
    hltParticleFlowRecHitECALUnseeded,
    hltParticleFlowSuperClusterECALUnseeded
)
