import FWCore.ParameterSet.Config as cms

from ..modules.hltFilteredLayerClustersCLUE3DBarrel_cfi import *
from ..modules.hltTiclSeedingGlobal_cfi import *
from ..modules.hltTiclTrackstersCLUE3DBarrel_cfi import *

HLTTiclTrackstersCLUE3DBarrelStepSequence = cms.Sequence(hltFilteredLayerClustersCLUE3DBarrel+hltTiclSeedingGlobal+hltTiclTrackstersCLUE3DBarrel)
