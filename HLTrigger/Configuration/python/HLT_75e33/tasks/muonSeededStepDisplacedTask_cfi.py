import FWCore.ParameterSet.Config as cms

from ..modules.earlyDisplacedMuons_cfi import *
from ..tasks.muonSeededStepCoreDisplacedTask_cfi import *
from ..tasks.muonSeededStepExtraDisplacedTask_cfi import *

muonSeededStepDisplacedTask = cms.Task(earlyDisplacedMuons, muonSeededStepCoreDisplacedTask, muonSeededStepExtraDisplacedTask)
