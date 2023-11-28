import FWCore.ParameterSet.Config as cms

from ..modules.hltHcalDigis_cfi import *
from ..modules.hgcalDigis_cfi import *
from ..modules.muonCSCDigis_cfi import *
from ..modules.muonDTDigis_cfi import *
from ..modules.muonGEMDigis_cfi import *
from ..tasks.hltEcalDigisTask_cfi import *

RawToDigiTask = cms.Task(
    hltEcalDigisTask,
    hltHcalDigis,
    hgcalDigis,
    muonCSCDigis,
    muonDTDigis,
    muonGEMDigis,
)
