import FWCore.ParameterSet.Config as cms

from ..tasks.HLTDoLocalHcalTask_cfi import *

HLTDoLocalHcalSequence = cms.Sequence(HLTDoLocalHcalTask)
