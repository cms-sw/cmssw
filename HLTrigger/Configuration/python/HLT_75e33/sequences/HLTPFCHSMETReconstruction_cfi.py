import FWCore.ParameterSet.Config as cms

from ..modules.hltParticleFlowCHS_cfi import *
from ..modules.hltPFCHSMET_cfi import *

HLTPFCHSMETReconstruction = cms.Sequence(hltParticleFlowCHS+hltPFCHSMET)
