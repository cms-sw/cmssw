import FWCore.ParameterSet.Config as cms

from ..modules.displacedMuonSeeds_cfi import *
from ..modules.displacedStandAloneMuons_cfi import *
from ..modules.refittedStandAloneMuons_cfi import *
from ..modules.standAloneMuons_cfi import *
from ..tasks.standAloneMuonSeedsTask_cfi import *

standalonemuontrackingTask = cms.Task(displacedMuonSeeds, displacedStandAloneMuons, refittedStandAloneMuons, standAloneMuonSeedsTask, standAloneMuons)
