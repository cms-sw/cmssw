import FWCore.ParameterSet.Config as cms

from ..modules.hltL1TEGammaFilteredCollectionProducer_cfi import *
from ..modules.hltParticleFlowClusterECALL1Seeded_cfi import *
from ..modules.hltParticleFlowClusterECALUncorrectedL1Seeded_cfi import *
from ..modules.hltParticleFlowClusterPSL1Seeded_cfi import *
from ..modules.hltParticleFlowRecHitECALL1Seeded_cfi import *
from ..modules.hltParticleFlowRecHitPSL1Seeded_cfi import *
from ..modules.hltParticleFlowSuperClusterECALL1Seeded_cfi import *
from ..modules.hltRechitInRegionsECAL_cfi import *

HLTPFClusteringForEgammaL1SeededTask = cms.Task(hltL1TEGammaFilteredCollectionProducer, hltParticleFlowClusterECALL1Seeded, hltParticleFlowClusterECALUncorrectedL1Seeded, hltParticleFlowClusterPSL1Seeded, hltParticleFlowRecHitECALL1Seeded, hltParticleFlowRecHitPSL1Seeded, hltParticleFlowSuperClusterECALL1Seeded, hltRechitInRegionsECAL)
