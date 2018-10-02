import FWCore.ParameterSet.Config as cms

#-------------------------------------------------
# Stage2 Emulator Modules (TODO: Move to L1Trigger.HardwareValidation.L1Stage2HardwareValidation_cff)

# Calo configuration
from L1Trigger.L1TCalorimeter.simDigis_cff import *
# CaloLayer1
from L1Trigger.L1TCaloLayer1.simCaloStage2Layer1Digis_cfi import simCaloStage2Layer1Digis
valCaloStage2Layer1Digis = simCaloStage2Layer1Digis.clone()
valCaloStage2Layer1Digis.ecalToken = cms.InputTag("caloLayer1Digis")
valCaloStage2Layer1Digis.hcalToken = cms.InputTag("caloLayer1Digis")
valCaloStage2Layer1Digis.unpackEcalMask = cms.bool(True)
valCaloStage2Layer1Digis.unpackHcalMask = cms.bool(True)

# CaloLayer2
from L1Trigger.L1TCalorimeter.simCaloStage2Digis_cfi import simCaloStage2Digis
valCaloStage2Layer2Digis = simCaloStage2Digis.clone()
valCaloStage2Layer2Digis.towerToken = cms.InputTag("caloStage2Digis", "CaloTower")

# BMTF-Legacy
from L1Trigger.L1TMuonBarrel.simBmtfDigis_cfi import *
valBmtfDigis = simBmtfDigis.clone()
valBmtfDigis.DTDigi_Source = cms.InputTag("bmtfDigis")
valBmtfDigis.DTDigi_Theta_Source = cms.InputTag("bmtfDigis")

# BMTF-Kalman
from L1Trigger.L1TMuonBarrel.simKBmtfDigis_cfi import *
from L1Trigger.L1TMuonBarrel.simKBmtfStubs_cfi import *
valKBmtfStubs = simKBmtfStubs.clone()
valKBmtfStubs.srcPhi = cms.InputTag("bmtfDigis")
valKBmtfStubs.srcTheta = cms.InputTag("bmtfDigis")
valKBmtfDigis = simKBmtfDigis.clone()
valKBmtfDigis.src = cms.InputTag("valKBmtfStubs")

# BMTF-AlgoTriggerSelector
from DQM.L1TMonitor.L1TBMTFAlgoSelector_cfi import *
valBmtfAlgoSel = l1tBmtfAlgoSelector.clone()
valBmtfAlgoSel.bmtfKalman = cms.InputTag("valKBmtfDigis:BMTF")
valBmtfAlgoSel.bmtfLegacy = cms.InputTag("valBmtfDigis:BMTF")

# OMTF
from L1Trigger.L1TMuonOverlap.simOmtfDigis_cfi import *
valOmtfDigis = simOmtfDigis.clone()
valOmtfDigis.srcDTPh = cms.InputTag('omtfStage2Digis')
valOmtfDigis.srcDTTh = cms.InputTag('omtfStage2Digis')
valOmtfDigis.srcCSC = cms.InputTag('omtfStage2Digis')
valOmtfDigis.srcRPC = cms.InputTag('omtfStage2Digis')

# EMTF
from L1Trigger.L1TMuonEndCap.simEmtfDigis_cfi import *
valEmtfStage2Digis = simEmtfDigis.clone()
valEmtfStage2Digis.CSCInput = "emtfStage2Digis"
valEmtfStage2Digis.RPCInput = "muonRPCDigis"

# uGMT
from L1Trigger.L1TMuon.simGmtStage2Digis_cfi import *
valGmtCaloSumDigis = simGmtCaloSumDigis.clone()
valGmtCaloSumDigis.caloStage2Layer2Label = cms.InputTag("valCaloStage2Layer1Digis")
valGmtStage2Digis = simGmtStage2Digis.clone()
valGmtStage2Digis.barrelTFInput = cms.InputTag("gmtStage2Digis", "BMTF")
valGmtStage2Digis.overlapTFInput = cms.InputTag("gmtStage2Digis", "OMTF")
valGmtStage2Digis.forwardTFInput = cms.InputTag("gmtStage2Digis", "EMTF")
valGmtStage2Digis.triggerTowerInput = cms.InputTag("valGmtCaloSumDigis", "TriggerTowerSums")

# uGT
from L1Trigger.L1TGlobal.simGtStage2Digis_cfi import simGtStage2Digis
from L1Trigger.L1TGlobal.simGtExtFakeProd_cfi import simGtExtFakeProd

valGtStage2Digis = simGtStage2Digis.clone()
valGtStage2Digis.ExtInputTag = cms.InputTag("gtStage2Digis")
valGtStage2Digis.MuonInputTag = cms.InputTag("gtStage2Digis", "Muon")
valGtStage2Digis.EGammaInputTag = cms.InputTag("gtStage2Digis", "EGamma")
valGtStage2Digis.TauInputTag = cms.InputTag("gtStage2Digis", "Tau")
valGtStage2Digis.JetInputTag = cms.InputTag("gtStage2Digis", "Jet")
valGtStage2Digis.EtSumInputTag = cms.InputTag("gtStage2Digis", "EtSum")
valGtStage2Digis.AlgorithmTriggersUnmasked = cms.bool(False)
valGtStage2Digis.AlgorithmTriggersUnprescaled = cms.bool(False)
valGtStage2Digis.EmulateBxInEvent = cms.int32(5)
valGtStage2Digis.PrescaleSet = cms.uint32(7)
valGtStage2Digis.GetPrescaleColumnFromData = cms.bool(True)
valGtStage2Digis.AlgoBlkInputTag = cms.InputTag("gtStage2Digis")

Stage2L1HardwareValidation = cms.Sequence(
    valCaloStage2Layer1Digis +
    valBmtfDigis +
    valKBmtfStubs +
    valKBmtfDigis +
    valBmtfAlgoSel +
    valOmtfDigis +
    valEmtfStage2Digis +
    valGmtCaloSumDigis +
    valGmtStage2Digis +
    valGtStage2Digis
)

Stage2L1HardwareValidationForValidationEvents = cms.Sequence(
    valCaloStage2Layer2Digis
)

#-------------------------------------------------
# Emulator DQM Modules

# CaloLayer1
from DQM.L1TMonitor.L1TdeStage2CaloLayer1_cfi import *

# CaloLayer2
from DQM.L1TMonitor.L1TdeStage2CaloLayer2_cfi import *
from DQM.L1TMonitor.L1TStage2CaloLayer2Emul_cfi import *

# BMTF
from DQM.L1TMonitor.L1TdeStage2BMTF_cfi import *
from DQM.L1TMonitor.L1TdeStage2BMTFSecond_cff import *

# OMTF
from DQM.L1TMonitor.L1TdeStage2OMTF_cfi import *

# EMTF
from DQM.L1TMonitor.L1TdeStage2EMTF_cff import *

# uGMT
from DQM.L1TMonitor.L1TdeStage2uGMT_cff import *

# uGT
from DQM.L1TMonitor.L1TStage2uGTEmul_cfi import *
from DQM.L1TMonitor.L1TdeStage2uGT_cfi import *

#-------------------------------------------------
# Stage2 Emulator and Emulator DQM Sequences

# sequence to run for every event
l1tStage2EmulatorOnlineDQM = cms.Sequence(
    l1tdeStage2Bmtf +
    l1tdeStage2BmtfSecond +
    l1tdeStage2Omtf +
    l1tdeStage2EmtfOnlineDQMSeq +
    l1tStage2uGMTEmulatorOnlineDQMSeq +
    l1tdeStage2uGT +
    l1tStage2uGtEmul
)

# sequence to run only for validation events
l1tStage2EmulatorOnlineDQMValidationEvents = cms.Sequence(
    l1tdeStage2CaloLayer1 +
    l1tdeStage2CaloLayer2 +
    l1tStage2CaloLayer2Emul
)
