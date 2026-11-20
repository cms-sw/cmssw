import FWCore.ParameterSet.Config as cms

from ..modules.l1tZDCEtSums_cfi import *
from ..modules.simBmtfDigis_cfi import *
from ..modules.simCaloStage2Digis_cfi import *
from ..modules.simCaloStage2Layer1Digis_cfi import *
from ..modules.simCaloStage2Layer1Summary_cfi import *
from ..modules.simCscTriggerPrimitiveDigisRun3_cfi import *
from ..modules.simEmtfDigis_cfi import *
from ..modules.simEmtfShowers_cfi import *
from ..modules.simGmtCaloSumDigis_cfi import *
from ..modules.simGmtShowerDigis_cfi import *
from ..modules.simGmtStage2Digis_cfi import *
from ..modules.simGtExtFakeStage2Digis_cfi import *
from ..modules.simGtStage2Digis_cfi import *
from ..modules.simKBmtfDigis_cfi import *
from ..modules.simKBmtfStubs_cfi import *
from ..modules.simOmtfDigis_cfi import *
from ..modules.simTwinMuxDigis_cfi import *

L1TLegacyRun3Task = cms.Task(l1tZDCEtSums, simBmtfDigis, simCaloStage2Digis, simCaloStage2Layer1Digis, simCaloStage2Layer1Summary, simCscTriggerPrimitiveDigisRun3, simEmtfDigis, simEmtfShowers, simGmtCaloSumDigis, simGmtShowerDigis, simGmtStage2Digis, simGtExtFakeStage2Digis, simGtStage2Digis, simKBmtfDigis, simKBmtfStubs, simOmtfDigis, simTwinMuxDigis)
