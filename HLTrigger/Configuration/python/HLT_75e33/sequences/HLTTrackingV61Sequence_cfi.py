import FWCore.ParameterSet.Config as cms

from ..tasks.HLTTrackingV61Task_cfi import *

HLTTrackingV61Sequence = cms.Sequence(HLTTrackingV61Task)
