import FWCore.ParameterSet.Config as cms

from ..modules.hltParticleFlowClusterHBHE_cfi import *
from ..modules.hltParticleFlowClusterHCAL_cfi import *
from ..modules.hltParticleFlowRecHitHBHE_cfi import *
from ..modules.particleFlowClusterECAL_cfi import *
from ..modules.particleFlowClusterECALUncorrected_cfi import *
from ..modules.particleFlowClusterHF_cfi import *
from ..modules.particleFlowClusterHO_cfi import *
from ..modules.particleFlowRecHitECAL_cfi import *
from ..modules.particleFlowRecHitHF_cfi import *
from ..modules.particleFlowRecHitHO_cfi import *
from ..modules.pfClusterRefsForJets_cfi import *
from ..modules.pfClusterRefsForJetsECAL_cfi import *
from ..modules.pfClusterRefsForJetsHCAL_cfi import *
from ..modules.pfClusterRefsForJetsHF_cfi import *
from ..modules.pfClusterRefsForJetsHGCAL_cfi import *
from ..modules.pfClusterRefsForJetsHO_cfi import *

pfClusterRefsForJetsSequence = cms.Sequence(particleFlowRecHitECAL+particleFlowRecHitHF+particleFlowRecHitHO+hltParticleFlowRecHitHBHE+hltParticleFlowClusterHBHE+hltParticleFlowClusterHCAL+particleFlowClusterECAL+particleFlowClusterECALUncorrected+particleFlowClusterHF+particleFlowClusterHO+pfClusterRefsForJetsECAL+pfClusterRefsForJetsHCAL+pfClusterRefsForJetsHF+pfClusterRefsForJetsHGCAL+pfClusterRefsForJetsHO+pfClusterRefsForJets)
