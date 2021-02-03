import FWCore.ParameterSet.Config as cms

from ..modules.displacedGlobalMuons_cfi import *
from ..tasks.iterDisplcedTrackingTask_cfi import *

displacedGlobalMuonTrackingTask = cms.Task(displacedGlobalMuons, iterDisplcedTrackingTask)
