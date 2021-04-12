import FWCore.ParameterSet.Config as cms

from ..tasks.localrecoTask_cfi import *

localreco = cms.Sequence(localrecoTask)
