import FWCore.ParameterSet.Config as cms

from ..tasks.hgcalLocalRecoTask_cfi import *

hgcalLocalRecoSequence = cms.Sequence(hgcalLocalRecoTask)
