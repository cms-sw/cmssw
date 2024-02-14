import FWCore.ParameterSet.Config as cms

from ..modules.filteredLayerClustersCLUE3DHigh_cfi import *
from ..modules.ticlSeedingGlobal_cfi import *
from ..modules.ticlTrackstersCLUE3DHigh_cfi import *

ticlTrackstersCLUE3DHighStepTask = cms.Task(filteredLayerClustersCLUE3DHigh, ticlSeedingGlobal, ticlTrackstersCLUE3DHigh)
