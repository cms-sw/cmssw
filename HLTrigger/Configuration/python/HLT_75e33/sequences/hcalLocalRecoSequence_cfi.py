import FWCore.ParameterSet.Config as cms

from ..tasks.hcalLocalRecoTask_cfi import *

hcalLocalRecoSequence = cms.Sequence(hcalLocalRecoTask)
