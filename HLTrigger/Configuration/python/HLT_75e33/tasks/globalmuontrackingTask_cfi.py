import FWCore.ParameterSet.Config as cms

from ..modules.globalMuons_cfi import *
from ..modules.tevMuons_cfi import *
from ..tasks.displacedGlobalMuonTrackingTask_cfi import *

globalmuontrackingTask = cms.Task(displacedGlobalMuonTrackingTask, globalMuons, tevMuons)
