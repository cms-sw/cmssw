import FWCore.ParameterSet.Config as cms

from ..modules.filteredLayerClustersEM_cfi import *
from ..modules.ticlSeedingGlobal_cfi import *
from ..modules.ticlTrackstersEM_cfi import *

ticlEMStepTask = cms.Task(filteredLayerClustersEM, ticlSeedingGlobal, ticlTrackstersEM)
