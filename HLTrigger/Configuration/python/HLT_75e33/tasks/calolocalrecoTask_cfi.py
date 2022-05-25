import FWCore.ParameterSet.Config as cms

from ..tasks.ecalLocalRecoTask_cfi import *
from ..tasks.hcalLocalRecoTask_cfi import *

calolocalrecoTask = cms.Task(
    ecalLocalRecoTask,
    hcalLocalRecoTask
)
