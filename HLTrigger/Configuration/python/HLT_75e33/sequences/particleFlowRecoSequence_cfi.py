import FWCore.ParameterSet.Config as cms

from ..modules.fixedGridRhoFastjetAllTmp_cfi import *
from ..modules.particleFlowBlock_cfi import *
from ..modules.particleFlowTmp_cfi import *
from ..modules.particleFlowTmpBarrel_cfi import *
from ..modules.pfTrack_cfi import *

particleFlowRecoSequence = cms.Sequence(pfTrack+particleFlowBlock+particleFlowTmpBarrel+particleFlowTmp+fixedGridRhoFastjetAllTmp)
