import FWCore.ParameterSet.Config as cms

from ..modules.filteredLayerClustersCLUE3DHigh_cfi import *
from ..modules.ticlSeedingGlobal_cfi import *
from ..modules.ticlTrackstersCLUE3DHigh_cfi import *

ticlTrackstersCLUE3DHighStepSequence = cms.Sequence(filteredLayerClustersCLUE3DHigh+ticlSeedingGlobal+ticlTrackstersCLUE3DHigh)
