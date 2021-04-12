import FWCore.ParameterSet.Config as cms

from ..tasks.calolocalrecoTask_cfi import *

calolocalreco = cms.Sequence(calolocalrecoTask)
