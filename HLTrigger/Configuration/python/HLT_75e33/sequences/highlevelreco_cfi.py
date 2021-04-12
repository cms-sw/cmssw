import FWCore.ParameterSet.Config as cms

from ..tasks.highlevelrecoTask_cfi import *

highlevelreco = cms.Sequence(highlevelrecoTask)
