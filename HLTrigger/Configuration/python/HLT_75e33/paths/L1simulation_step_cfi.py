import FWCore.ParameterSet.Config as cms

from ..modules.simBscDigis_cfi import *
from ..modules.simCastorTechTrigDigis_cfi import *
from ..modules.simCsctfDigis_cfi import *
from ..modules.simCsctfTrackDigis_cfi import *
from ..modules.simCscTriggerPrimitiveDigis_cfi import *
from ..modules.simDttfDigis_cfi import *
from ..modules.simDtTriggerPrimitiveDigis_cfi import *
from ..modules.simGctDigis_cfi import *
from ..modules.simGmtDigis_cfi import *
from ..modules.simGtDigis_cfi import *
from ..modules.simHcalTechTrigDigis_cfi import *
from ..modules.simRctDigis_cfi import *
from ..modules.simRpcTechTrigDigis_cfi import *
from ..modules.simRpcTriggerDigis_cfi import *

L1simulation_step = cms.Path(cms.Task(simBscDigis, simCastorTechTrigDigis, simCscTriggerPrimitiveDigis, simCsctfDigis, simCsctfTrackDigis, simDtTriggerPrimitiveDigis, simDttfDigis, simGctDigis, simGmtDigis, simGtDigis, simHcalTechTrigDigis, simRctDigis, simRpcTechTrigDigis, simRpcTriggerDigis))
