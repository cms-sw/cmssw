import FWCore.ParameterSet.Config as cms

from ..tasks.globalrecoTask_cfi import *

globalreco = cms.Sequence(globalrecoTask)
