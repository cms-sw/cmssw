import FWCore.ParameterSet.Config as cms

from ..tasks.globalmuontrackingTask_cfi import *
from ..tasks.muIsolation_muonsTask_cfi import *
from ..tasks.muonIdProducerTask_cfi import *

muonGlobalRecoTask = cms.Task(globalmuontrackingTask, muIsolation_muonsTask, muonIdProducerTask)
