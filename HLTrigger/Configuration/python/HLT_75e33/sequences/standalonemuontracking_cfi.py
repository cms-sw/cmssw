import FWCore.ParameterSet.Config as cms

from ..tasks.standalonemuontrackingTask_cfi import *

standalonemuontracking = cms.Sequence(standalonemuontrackingTask)
