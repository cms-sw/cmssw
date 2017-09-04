import FWCore.ParameterSet.Config as cms

#-------------------------------------------------
# Stage2 Unpacker Modules
# TODO: This needs to be setup as a StandardSequence.

# CaloLayer1
from EventFilter.L1TXRawToDigi.caloLayer1Stage2Digis_cfi import *

# CaloLayer2
from EventFilter.L1TRawToDigi.caloStage2Digis_cfi import *

# BMTF 
from EventFilter.L1TRawToDigi.bmtfDigis_cfi import *

# OMTF
#from EventFilter.L1TRawToDigi.omtfStage2Digis_cfi import *

# EMTF
from EventFilter.L1TRawToDigi.emtfStage2Digis_cfi import *

# uGMT
from EventFilter.L1TRawToDigi.gmtStage2Digis_cfi import *

# uGT
from EventFilter.L1TRawToDigi.gtStage2Digis_cfi import *

l1tStage2Unpack = cms.Sequence(
    l1tCaloLayer1Digis +
    gmtStage2Digis
)

l1tStage2UnpackValidationEvents = cms.Sequence(
    caloStage2Digis +
    bmtfDigis  +
    #omtfStage2Digis +
    emtfStage2Digis +
    gtStage2Digis
)

#-------------------------------------------------
# Stage2 Emulator Modules (TODO: Move to L1Trigger.HardwareValidation.L1Stage2HardwareValidation_cff)

# CaloLayer1
from L1Trigger.L1TCaloLayer1.simCaloStage2Layer1Digis_cfi import simCaloStage2Layer1Digis
valCaloStage2Layer1Digis = simCaloStage2Layer1Digis.clone()
valCaloStage2Layer1Digis.ecalToken = cms.InputTag("l1tCaloLayer1Digis")
valCaloStage2Layer1Digis.hcalToken = cms.InputTag("l1tCaloLayer1Digis")
valCaloStage2Layer1Digis.unpackEcalMask = cms.bool(True)
valCaloStage2Layer1Digis.unpackHcalMask = cms.bool(True)

# CaloLayer2
from L1Trigger.L1TCalorimeter.simCaloStage2Digis_cfi import simCaloStage2Digis
valCaloStage2Layer2Digis = simCaloStage2Digis.clone()
valCaloStage2Layer2Digis.towerToken = cms.InputTag("caloStage2Digis", "CaloTower")

# BMTF
from L1Trigger.L1TMuonBarrel.simBmtfDigis_cfi import *
valBmtfDigis = simBmtfDigis.clone()
valBmtfDigis.DTDigi_Source = cms.InputTag("bmtfDigis")
valBmtfDigis.DTDigi_Theta_Source = cms.InputTag("bmtfDigis")

# OMTF
from L1Trigger.L1TMuonOverlap.simOmtfDigis_cfi import *
valOmtfDigis = simOmtfDigis.clone()
valOmtfDigis.srcDTPh = cms.InputTag('bmtfDigis')
valOmtfDigis.srcDTTh = cms.InputTag('bmtfDigis')
valOmtfDigis.srcCSC = cms.InputTag('emtfStage2Digis')
valOmtfDigis.srcRPC = cms.InputTag('muonRPCDigis')

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
valGtStage2Digis.ExtInputTag = cms.InputTag("simGtExtFakeProd")
valGtStage2Digis.MuonInputTag = cms.InputTag("gmtStage2Digis", "Muon")
valGtStage2Digis.EGammaInputTag = cms.InputTag("caloStage2Digis", "EGamma")
valGtStage2Digis.TauInputTag = cms.InputTag("caloStage2Digis", "Tau")
valGtStage2Digis.JetInputTag = cms.InputTag("caloStage2Digis", "Jet")
valGtStage2Digis.EtSumInputTag = cms.InputTag("caloStage2Digis", "EtSum")
valGtStage2Digis.AlgorithmTriggersUnmasked = cms.bool(False)
valGtStage2Digis.AlgorithmTriggersUnprescaled = cms.bool(False)

Stage2L1HardwareValidation = cms.Sequence(
    valCaloStage2Layer1Digis +
    valGmtCaloSumDigis +
    valGmtStage2Digis
)

Stage2L1HardwareValidationForValidationEvents = cms.Sequence(
    valCaloStage2Layer2Digis +
    valBmtfDigis +
    valEmtfStage2Digis +
    valOmtfDigis +
    valGtStage2Digis
)

#-------------------------------------------------
# Emulator DQM Modules

# CaloLayer1
from DQM.L1TMonitor.L1TdeStage2CaloLayer1_cfi import *

# CaloLayer2
from DQM.L1TMonitor.L1TdeStage2CaloLayer2_cfi import *
from DQM.L1TMonitor.L1TStage2CaloLayer2_cfi import *
from DQM.L1TMonitor.L1TStage2CaloLayer2Emul_cfi import *

# BMTF
from DQM.L1TMonitor.L1TdeStage2BMTF_cfi import *

# OMTF
from DQM.L1TMonitor.L1TdeStage2OMTF_cfi import *

# EMTF
from DQM.L1TMonitor.L1TdeStage2EMTF_cff import *

# uGMT
from DQM.L1TMonitor.L1TdeStage2uGMT_cff import *

# uGT
from DQM.L1TMonitor.L1TStage2uGTEmul_cfi import *

#-------------------------------------------------
# Stage2 Emulator and Emulator DQM Sequences

# sequence to run for every event
l1tStage2EmulatorOnlineDQM = cms.Sequence(
    l1tStage2uGMTEmulatorOnlineDQMSeq
)

# sequence to run only for validation events
l1tStage2EmulatorOnlineDQMValidationEvents = cms.Sequence(
    l1tdeStage2CaloLayer1 +
    # We process both layer2 and layer2emu in same sourceclient
    # to be able to divide them in the MonitorClient
    l1tdeStage2CaloLayer2 +
    l1tStage2CaloLayer2 + l1tStage2CaloLayer2Emul +
    l1tdeStage2Bmtf +
    l1tdeStage2Omtf +
    l1tdeStage2Emtf +
    l1tdeStage2EmtfComp +
    l1tStage2uGtEmul
)

