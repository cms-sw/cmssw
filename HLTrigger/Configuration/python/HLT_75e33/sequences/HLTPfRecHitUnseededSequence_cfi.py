import FWCore.ParameterSet.Config as cms

from ..modules.hltParticleFlowRecHitECALUnseeded_cfi import *
from ..modules.hltParticleFlowRecHitHBHE_cfi import *

HLTPfRecHitUnseededSequence = cms.Sequence(hltParticleFlowRecHitECALUnseeded+hltParticleFlowRecHitHBHE)
