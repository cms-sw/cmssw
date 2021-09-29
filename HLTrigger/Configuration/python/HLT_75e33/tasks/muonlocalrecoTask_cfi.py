import FWCore.ParameterSet.Config as cms

from ..modules.rpcRecHits_cfi import *
from ..tasks.csclocalrecoTask_cfi import *
from ..tasks.dtlocalrecoTask_cfi import *
from ..tasks.gemLocalRecoTask_cfi import *
from ..tasks.me0LocalRecoTask_cfi import *

muonlocalrecoTask = cms.Task(
    csclocalrecoTask,
    dtlocalrecoTask,
    gemLocalRecoTask,
    me0LocalRecoTask,
    rpcRecHits
)
