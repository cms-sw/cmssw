import FWCore.ParameterSet.Config as cms

from ..modules.caloStage1Digis_cfi import *
from ..modules.caloStage1FinalDigis_cfi import *
from ..modules.caloStage1LegacyFormatDigis_cfi import *
from ..modules.csctfDigis_cfi import *
from ..modules.dttfDigis_cfi import *
from ..modules.gctDigis_cfi import *
from ..modules.gtDigis_cfi import *

L1TRawToDigi_Stage1 = cms.Task(caloStage1Digis, caloStage1FinalDigis, caloStage1LegacyFormatDigis, csctfDigis, dttfDigis, gctDigis, gtDigis)
