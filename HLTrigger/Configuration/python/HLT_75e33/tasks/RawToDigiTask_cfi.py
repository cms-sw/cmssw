import FWCore.ParameterSet.Config as cms

from ..modules.ecalPreshowerDigis_cfi import *
from ..modules.hcalDigis_cfi import *
from ..modules.hgcalDigis_cfi import *
from ..modules.muonCSCDigis_cfi import *
from ..modules.muonDTDigis_cfi import *
from ..modules.muonGEMDigis_cfi import *
from ..modules.muonRPCDigis_cfi import *
from ..modules.onlineMetaDataDigis_cfi import *
from ..modules.scalersRawToDigi_cfi import *
from ..modules.siStripDigis_cfi import *
from ..modules.tcdsDigis_cfi import *
from ..tasks.ctppsRawToDigiTask_cfi import *
from ..tasks.ecalDigisTask_cfi import *
from ..tasks.L1TRawToDigiTask_cfi import *

RawToDigiTask = cms.Task(L1TRawToDigiTask, ctppsRawToDigiTask, ecalDigisTask, ecalPreshowerDigis, hcalDigis, hgcalDigis, muonCSCDigis, muonDTDigis, muonGEMDigis, muonRPCDigis, onlineMetaDataDigis, scalersRawToDigi, siStripDigis, tcdsDigis)
