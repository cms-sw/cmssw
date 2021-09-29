import FWCore.ParameterSet.Config as cms

from ..modules.ecalPreshowerDigis_cfi import *
from ..modules.hcalDigis_cfi import *
from ..modules.hgcalDigis_cfi import *
from ..modules.muonCSCDigis_cfi import *
from ..modules.muonDTDigis_cfi import *
from ..modules.muonGEMDigis_cfi import *
from ..modules.siStripDigis_cfi import *
from ..tasks.ecalDigisTask_cfi import *

RawToDigiTask = cms.Task(
    ecalDigisTask,
    ecalPreshowerDigis,
    hcalDigis,
    hgcalDigis,
    muonCSCDigis,
    muonDTDigis,
    muonGEMDigis,
    siStripDigis
)
