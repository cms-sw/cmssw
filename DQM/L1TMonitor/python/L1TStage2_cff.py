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
from EventFilter.L1TRawToDigi.omtfStage2Digis_cfi import *

# EMTF
from EventFilter.L1TRawToDigi.emtfStage2Digis_cfi import *

# uGMT
from EventFilter.L1TRawToDigi.gmtStage2Digis_cfi import *

# uGT
from EventFilter.L1TRawToDigi.gtStage2Digis_cfi import *

l1tStage2Unpack = cms.Sequence(
    l1tCaloLayer1Digis +
    caloStage2Digis +
    bmtfDigis  +
    omtfStage2Digis +
    emtfStage2Digis +
    gmtStage2Digis +
    gtStage2Digis
)

#-------------------------------------------------
# DQM Modules

# CaloLayer1
from DQM.L1TMonitor.L1TStage2CaloLayer1_cfi import *

# CaloLayer2
# Since layer2 and layer2 emulation are to be divided by each other
# in the l1tstage2emulator sourceclient, we process
# stage 2 occupancy in that client rather than this one

# UPDATE Apr 21: Since emulator client is stalled due to 
# GlobalTag/CaloParams conflicts, we move back to this client
from DQM.L1TMonitor.L1TStage2CaloLayer2_cfi import *
from DQM.L1TMonitor.L1TStage2uGTCaloLayer2Comp_cfi import *

# BMTF
from DQM.L1TMonitor.L1TStage2BMTF_cfi import *

# OMTF
from DQM.L1TMonitor.L1TStage2OMTF_cfi import *

# EMTF
from DQM.L1TMonitor.L1TStage2EMTF_cfi import *

# uGMT
from DQM.L1TMonitor.L1TStage2uGMT_cff import *

# uGT
from DQM.L1TMonitor.L1TStage2uGT_cfi import *

#-------------------------------------------------
# Stage2 Unpacking and DQM Sequences

# sequence to run for every event
l1tStage2OnlineDQM = cms.Sequence(
    l1tStage2CaloLayer1  +
    l1tStage2CaloLayer2 +
    l1tStage2Bmtf +
    l1tStage2Omtf +
    l1tStage2Emtf +
    l1tStage2uGMTOnlineDQMSeq +
    l1tStage2uGTCaloLayer2Comp +
    l1tStage2uGt
)

# sequence to run only for validation events
l1tStage2OnlineDQMValidationEvents = cms.Sequence(
    l1tStage2uGMTValidationEventOnlineDQMSeq
)

