#
# WARNING: This file is in the L1T configuration critical path.
#
# All changes must be explicitly discussed with the L1T offline coordinator.
#
import FWCore.ParameterSet.Config as cms
from L1Trigger.L1TGlobal.simGtStage2DigisDef_cfi import simGtStage2DigisDef
simGtStage2Digis = simGtStage2DigisDef.clone(
    MuonInputTag = "simGmtStage2Digis",
    MuonShowerInputTag = "simGmtShowerDigis",
    EGammaInputTag = "simCaloStage2Digis",
    TauInputTag = "simCaloStage2Digis",
    JetInputTag = "simCaloStage2Digis",
    EtSumInputTag = "simCaloStage2Digis",
    ExtInputTag = "simGtExtFakeStage2Digis",
    AlgoBlkInputTag = "gtStage2Digis",
    AlgorithmTriggersUnmasked = True,
    AlgorithmTriggersUnprescaled = True,
    GetPrescaleColumnFromData = False,
    RequireMenuToMatchAlgoBlkInput = False,
)

from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(simGtStage2Digis,
                     useMuonShowers = False)
