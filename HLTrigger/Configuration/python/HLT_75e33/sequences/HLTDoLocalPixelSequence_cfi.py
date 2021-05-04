import FWCore.ParameterSet.Config as cms

from ..tasks.HLTDoLocalPixelTask_cfi import *

HLTDoLocalPixelSequence = cms.Sequence(
    HLTDoLocalPixelTask
)
