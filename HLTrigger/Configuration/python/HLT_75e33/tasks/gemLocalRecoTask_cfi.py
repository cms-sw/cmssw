import FWCore.ParameterSet.Config as cms

from ..modules.gemRecHits_cfi import *
from ..modules.gemSegments_cfi import *

gemLocalRecoTask = cms.Task(gemRecHits, gemSegments)
