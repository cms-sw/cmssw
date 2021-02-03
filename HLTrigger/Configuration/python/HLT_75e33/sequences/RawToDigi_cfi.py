import FWCore.ParameterSet.Config as cms

from ..tasks.RawToDigiTask_cfi import *

RawToDigi = cms.Sequence(RawToDigiTask)
