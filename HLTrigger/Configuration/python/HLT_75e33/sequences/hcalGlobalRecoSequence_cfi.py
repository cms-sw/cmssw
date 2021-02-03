import FWCore.ParameterSet.Config as cms

from ..tasks.hcalGlobalRecoTask_cfi import *

hcalGlobalRecoSequence = cms.Sequence(hcalGlobalRecoTask)
