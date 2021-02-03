import FWCore.ParameterSet.Config as cms

from ..tasks.trackerlocalrecoTask_cfi import *

trackerlocalreco = cms.Sequence(trackerlocalrecoTask)
