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

# EMTF
from L1Trigger.L1TMuonEndCap.simEmtfDigis_cfi import *
valEmtfStage2Digis = simEmtfDigis.clone()
valEmtfStage2Digis.CSCInput = "csctfDigis"

Stage2L1HardwareValidation = cms.Sequence(
    valCaloStage2Layer1Digis +
    valEmtfStage2Digis
)

#-------------------------------------------------
# Emulator DQM Modules

# CaloLayer1
from DQM.L1TMonitor.L1TdeStage2CaloLayer1_cfi import *

# EMTF
from DQM.L1TMonitor.L1TdeStage2EMTF_cfi import *

#-------------------------------------------------
# Stage2 Emulator and Emulator DQM Sequences

l1tStage2EmulatorOnlineDQM = cms.Sequence(
    l1tdeStage2CaloLayer1 +
    l1tdeStage2Emtf
)

