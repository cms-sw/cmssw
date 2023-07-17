import FWCore.ParameterSet.Config as cms

from L1Trigger.L1CaloTrigger.phase2L1CaloEGammaEmulator_cfi import phase2L1CaloEGammaEmulator
l1tPhase2L1CaloEGammaEmulator = phase2L1CaloEGammaEmulator.clone()

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(l1tPhase2L1CaloEGammaEmulator,
    ecalTPEB = "DMEcalEBTriggerPrimitiveDigis",
    hcalTP = "DMHcalTriggerPrimitiveDigis",
)
