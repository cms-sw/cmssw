import FWCore.ParameterSet.Config as cms

#-------------------------------------------------
# Stage2 Unpacker Modules
# TODO: This needs to be setup as a StandardSequence.

# CaloLayer1
from EventFilter.L1TXRawToDigi.caloLayer1Stage2Digis_cfi import *

# CaloLayer2
from EventFilter.L1TRawToDigi.caloStage2Digis_cfi import *

# BMTF 
from EventFilter.L1TRawToDigi.l1tRawtoDigiBMTF_cfi import *

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
    caloStage2Digis +
    BMTFStage2Digis +
    #omtfStage2Digis +
    emtfStage2Digis +
    gmtStage2Digis +
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

# EMTF
from L1Trigger.L1TMuonEndCap.simEmtfDigis_cfi import *
valEmtfStage2Digis = simEmtfDigis.clone()
valEmtfStage2Digis.CSCInput = "csctfDigis"

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
    valCaloStage2Layer2Digis +
    valEmtfStage2Digis +
    valGtStage2Digis
)

#-------------------------------------------------
# Emulator DQM Modules

# CaloLayer1
from DQM.L1TMonitor.L1TdeStage2CaloLayer1_cfi import *

# CaloLayer2
from DQM.L1TMonitor.L1TStage2CaloLayer2_cfi import *
from DQM.L1TMonitor.L1TStage2CaloLayer2Emul_cfi import *

# EMTF
from DQM.L1TMonitor.L1TdeStage2EMTF_cfi import *

# uGT
from DQM.L1TMonitor.L1TStage2uGTEmul_cfi import *

#-------------------------------------------------
# Stage2 Emulator and Emulator DQM Sequences

l1tStage2EmulatorOnlineDQM = cms.Sequence(
    l1tdeStage2CaloLayer1 +
    # We process both layer2 and layer2emu in same sourceclient
    # to be able to divide them in the MonitorClient
    l1tStage2CaloLayer2 + l1tStage2CaloLayer2Emul +
    l1tdeStage2Emtf +
    l1tStage2uGtEmul
)
