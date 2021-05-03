import FWCore.ParameterSet.Config as cms

from ..tasks.HLTDoLocalStripTask_cfi import *

HLTDoLocalStripSequence = cms.Sequence(HLTDoLocalStripTask)
