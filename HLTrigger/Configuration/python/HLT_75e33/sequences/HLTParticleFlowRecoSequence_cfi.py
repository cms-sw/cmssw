import FWCore.ParameterSet.Config as cms

from ..modules.hltFixedGridRhoFastjetAll_cfi import *
from ..modules.hltParticleFlowBlock_cfi import *
from ..modules.hltParticleFlowTmp_cfi import *
from ..modules.hltParticleFlowTmpBarrel_cfi import *
from ..modules.hltPfTrack_cfi import *

HLTParticleFlowRecoSequence = cms.Sequence(hltPfTrack+hltParticleFlowBlock+hltParticleFlowTmpBarrel+hltParticleFlowTmp+hltFixedGridRhoFastjetAll)
