import FWCore.ParameterSet.Config as cms

from ..tasks.ecalLocalRecoTask_cfi import *

ecalLocalRecoSequence = cms.Sequence(ecalLocalRecoTask)
