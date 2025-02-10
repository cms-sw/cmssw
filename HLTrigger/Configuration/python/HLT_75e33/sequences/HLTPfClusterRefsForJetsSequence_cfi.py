import FWCore.ParameterSet.Config as cms

from ..modules.hltParticleFlowClusterHBHE_cfi import *
from ..modules.hltParticleFlowClusterHCAL_cfi import *
from ..modules.hltParticleFlowRecHitHBHE_cfi import *
from ..modules.hltParticleFlowClusterECAL_cfi import *
from ..modules.hltParticleFlowClusterECALUncorrected_cfi import *
from ..modules.hltParticleFlowClusterHF_cfi import *
from ..modules.hltParticleFlowClusterHO_cfi import *
from ..modules.hltParticleFlowRecHitECALUnseeded_cfi import *
from ..modules.hltParticleFlowRecHitHF_cfi import *
from ..modules.hltParticleFlowRecHitHO_cfi import *
from ..modules.hltPfClusterRefsForJets_cfi import *
from ..modules.hltPfClusterRefsForJetsECAL_cfi import *
from ..modules.hltPfClusterRefsForJetsHCAL_cfi import *
from ..modules.hltPfClusterRefsForJetsHF_cfi import *
from ..modules.hltPfClusterRefsForJetsHGCAL_cfi import *
from ..modules.hltPfClusterRefsForJetsHO_cfi import *

HLTPfClusterRefsForJetsSequence = cms.Sequence(hltParticleFlowRecHitECALUnseeded+
                                               hltParticleFlowRecHitHF+
                                               hltParticleFlowRecHitHO+
                                               hltParticleFlowRecHitHBHE+
                                               hltParticleFlowClusterHBHE+
                                               hltParticleFlowClusterHCAL+
                                               hltParticleFlowClusterECAL+
                                               hltParticleFlowClusterECALUncorrected+
                                               hltParticleFlowClusterHF+
                                               hltParticleFlowClusterHO+
                                               hltPfClusterRefsForJetsECAL+
                                               hltPfClusterRefsForJetsHCAL+
                                               hltPfClusterRefsForJetsHF+
                                               hltPfClusterRefsForJetsHGCAL+
                                               hltPfClusterRefsForJetsHO+
                                               hltPfClusterRefsForJets)
