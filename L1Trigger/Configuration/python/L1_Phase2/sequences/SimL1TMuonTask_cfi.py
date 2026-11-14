import FWCore.ParameterSet.Config as cms

from ..sequences.SimL1TMuonCommonTask_cfi import *
from ..sequences.me0TriggerAllDigiTask_cfi import *
from ..modules.simBmtfDigis_cfi import *
from ..modules.simCscTriggerPrimitiveDigisPhase2_cfi import *
from ..modules.simCscTriggerPrimitiveDigisRun3_cfi import *
from ..modules.simEmtfDigis_cfi import *
from ..modules.simEmtfShowers_cfi import *
from ..modules.simGmtCaloSumDigis_cfi import *
from ..modules.simGmtShowerDigis_cfi import *
from ..modules.simGmtStage2Digis_cfi import *
from ..modules.simKBmtfDigis_cfi import *
from ..modules.simKBmtfStubs_cfi import *
from ..sequences.simMuonGEMPadTask_cfi import *
from ..modules.simOmtfDigis_cfi import *
from ..modules.simTwinMuxDigis_cfi import *

SimL1TMuonTask = cms.Task(SimL1TMuonCommonTask, me0TriggerAllDigiTask, simBmtfDigis, simCscTriggerPrimitiveDigisPhase2, simCscTriggerPrimitiveDigisRun3, simEmtfDigis, simEmtfShowers, simGmtCaloSumDigis, simGmtShowerDigis, simGmtStage2Digis, simKBmtfDigis, simKBmtfStubs, simMuonGEMPadTask, simOmtfDigis, simTwinMuxDigis)
