import FWCore.ParameterSet.Config as cms

from ..modules.particleFlowClusterECAL_cfi import *
from ..modules.particleFlowClusterECALUncorrected_cfi import *
from ..modules.particleFlowClusterHBHE_cfi import *
from ..modules.particleFlowClusterHCAL_cfi import *
from ..modules.particleFlowClusterHF_cfi import *
from ..modules.particleFlowClusterHO_cfi import *
from ..modules.particleFlowRecHitECAL_cfi import *
from ..modules.particleFlowRecHitHBHE_cfi import *
from ..modules.particleFlowRecHitHF_cfi import *
from ..modules.particleFlowRecHitHO_cfi import *
from ..modules.pfClusterRefsForJets_cfi import *
from ..modules.pfClusterRefsForJetsECAL_cfi import *
from ..modules.pfClusterRefsForJetsHCAL_cfi import *
from ..modules.pfClusterRefsForJetsHF_cfi import *
from ..modules.pfClusterRefsForJetsHGCAL_cfi import *
from ..modules.pfClusterRefsForJetsHO_cfi import *

pfClusterRefsForJets_stepTask = cms.Task(particleFlowClusterECAL, particleFlowClusterECALUncorrected, particleFlowClusterHBHE, particleFlowClusterHCAL, particleFlowClusterHF, particleFlowClusterHO, particleFlowRecHitECAL, particleFlowRecHitHBHE, particleFlowRecHitHF, particleFlowRecHitHO, pfClusterRefsForJets, pfClusterRefsForJetsECAL, pfClusterRefsForJetsHCAL, pfClusterRefsForJetsHF, pfClusterRefsForJetsHGCAL, pfClusterRefsForJetsHO)
