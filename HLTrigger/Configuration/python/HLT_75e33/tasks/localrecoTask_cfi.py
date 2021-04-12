import FWCore.ParameterSet.Config as cms

from ..modules.bunchSpacingProducer_cfi import *
from ..tasks.calolocalrecoTask_cfi import *
from ..tasks.muonlocalrecoTask_cfi import *
from ..tasks.trackerlocalrecoTask_cfi import *

localrecoTask = cms.Task(bunchSpacingProducer, calolocalrecoTask, muonlocalrecoTask, trackerlocalrecoTask)
