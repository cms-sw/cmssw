import FWCore.ParameterSet.Config as cms

from ..modules.hltParticleFlowClusterECALUncorrectedUnseeded_cfi import *
from ..modules.hltParticleFlowClusterECALUnseeded_cfi import *
from ..modules.hltParticleFlowClusterPSUnseeded_cfi import *
from ..modules.hltParticleFlowRecHitECALUnseeded_cfi import *
from ..modules.hltParticleFlowRecHitPSUnseeded_cfi import *
from ..modules.hltParticleFlowSuperClusterECALUnseeded_cfi import *

HLTPFClusteringForEgammaUnseededTask = cms.Task(hltParticleFlowClusterECALUncorrectedUnseeded, hltParticleFlowClusterECALUnseeded, hltParticleFlowClusterPSUnseeded, hltParticleFlowRecHitECALUnseeded, hltParticleFlowRecHitPSUnseeded, hltParticleFlowSuperClusterECALUnseeded)
