from __future__ import print_function
import FWCore.ParameterSet.Config as cms

## L1REPACK CalouGT : Re-Emulate all of L1 and repack into RAW

from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger

(~stage2L1Trigger).toModify(None, lambda x:
    print("# L1T WARN:  L1REPACK:CalouGT (intended for 2016/2017 data) only supports Stage 2 eras for now.\n# L1T WARN:  Use a legacy version of L1REPACK for now."))
stage2L1Trigger.toModify(None, lambda x:
    print("# L1T INFO:  L1REPACK:CalouGT (intended for 2016/2017 data), reemulates the Calo part, uses unpacked Muons, and reemulates uGT."))

# First, Unpack all inputs to L1:
import EventFilter.L1TRawToDigi.bmtfDigis_cfi
unpackBmtf = EventFilter.L1TRawToDigi.bmtfDigis_cfi.bmtfDigis.clone(
    InputLabel = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess()))

import EventFilter.DTTFRawToDigi.dttfunpacker_cfi
unpackDttf = EventFilter.DTTFRawToDigi.dttfunpacker_cfi.dttfunpacker.clone(
    DTTF_FED_Source = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess()))
        
import EventFilter.L1TRawToDigi.emtfStage2Digis_cfi
unpackEmtf = EventFilter.L1TRawToDigi.emtfStage2Digis_cfi.emtfStage2Digis.clone(
    InputLabel = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess()))
        
import EventFilter.CSCTFRawToDigi.csctfunpacker_cfi
unpackCsctf = EventFilter.CSCTFRawToDigi.csctfunpacker_cfi.csctfunpacker.clone(
    producer = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess()))

import EventFilter.CSCRawToDigi.cscUnpacker_cfi
unpackCSC = EventFilter.CSCRawToDigi.cscUnpacker_cfi.muonCSCDigis.clone(
    InputObjects = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess()))

import EventFilter.DTRawToDigi.dtunpacker_cfi
unpackDT = EventFilter.DTRawToDigi.dtunpacker_cfi.muonDTDigis.clone(
    inputLabel = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess()))

import EventFilter.RPCRawToDigi.rpcUnpacker_cfi
unpackRPC = EventFilter.RPCRawToDigi.rpcUnpacker_cfi.rpcunpacker.clone(
    InputLabel = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess()))

import EventFilter.EcalRawToDigi.EcalUnpackerData_cfi
unpackEcal = EventFilter.EcalRawToDigi.EcalUnpackerData_cfi.ecalEBunpacker.clone(
    InputLabel = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess()))

import EventFilter.HcalRawToDigi.HcalRawToDigi_cfi
unpackHcal = EventFilter.HcalRawToDigi.HcalRawToDigi_cfi.hcalDigis.clone(
    InputLabel = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess()))

# Second, unpacker of inputs to uGT:
# #################################
import EventFilter.L1TRawToDigi.gtStage2Digis_cfi
unpackGtStage2 = EventFilter.L1TRawToDigi.gtStage2Digis_cfi.gtStage2Digis.clone(
    InputLabel = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess()))

# For simplicity, re-emulate the entire L1T (will not use Muon part)
# ################################################################

from SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff import *
simHcalTriggerPrimitiveDigis.inputLabel = [
    'unpackHcal',
    'unpackHcal'
]
simHcalTriggerPrimitiveDigis.inputUpgradeLabel = [
    'unpackHcal',     # upgrade HBHE
    'unpackHcal'      # upgrade HF
]

from L1Trigger.Configuration.SimL1Emulator_cff import *
    
simDtTriggerPrimitiveDigis.digiTag = 'unpackDT'
simCscTriggerPrimitiveDigis.CSCComparatorDigiProducer = 'unpackCSC:MuonCSCComparatorDigi'
simCscTriggerPrimitiveDigis.CSCWireDigiProducer       = 'unpackCSC:MuonCSCWireDigi'

simTwinMuxDigis.RPC_Source         = 'unpackRPC'
simTwinMuxDigis.DTDigi_Source      = "simDtTriggerPrimitiveDigis"
simTwinMuxDigis.DTThetaDigi_Source = "simDtTriggerPrimitiveDigis"

# -----------------------------------------------------------
# change when availalbe simTwinMux and reliable DTTPs, CSCTPs
cutlist=['simDtTriggerPrimitiveDigis','simCscTriggerPrimitiveDigis','simTwinMuxDigis']
for b in cutlist:
    SimL1EmulatorCore.remove(b)
# -----------------------------------------------------------

# BMTF
simBmtfDigis.DTDigi_Source       = "unpackBmtf"
simBmtfDigis.DTDigi_Theta_Source = "unpackBmtf"

# OMTF
simOmtfDigis.srcRPC              = 'unpackRPC'
simOmtfDigis.srcDTPh             = "unpackBmtf"
simOmtfDigis.srcDTTh             = "unpackBmtf"
simOmtfDigis.srcCSC              = "unpackCsctf" ## Replace when emtfStage2Digis give equal data-emulator agreement

# EMTF
simEmtfDigis.CSCInput            = "unpackEmtf"
simEmtfDigis.RPCInput            = 'unpackRPC'

simCaloStage2Layer1Digis.ecalToken = 'unpackEcal:EcalTriggerPrimitives'
simCaloStage2Layer1Digis.hcalToken = 'unpackHcal'

# uGT inputs for Muons are from unpacked
simGtStage2Digis.MuonInputTag   = "unpackGtStage2:Muon"

# Finally, pack the newly re-emulated L1T parts back into RAW
# Calo packer
from EventFilter.L1TRawToDigi.caloStage2Raw_cfi import caloStage2Raw as packCaloStage2
# uGT packer
from EventFilter.L1TRawToDigi.gtStage2Raw_cfi import gtStage2Raw as packGtStage2

# combine the new L1 RAW with existing RAW for other FEDs
import EventFilter.RawDataCollector.rawDataCollectorByLabel_cfi
rawDataCollector = EventFilter.RawDataCollector.rawDataCollectorByLabel_cfi.rawDataCollector.clone(
    verbose = 0,
    RawCollectionList = [
        'packCaloStage2',
        'packGtStage2',
        cms.InputTag('rawDataCollector', processName=cms.InputTag.skipCurrentProcess()),
        ]
    )


SimL1EmulatorTask = cms.Task()
stage2L1Trigger.toReplaceWith(SimL1EmulatorTask, cms.Task(unpackEcal,unpackHcal,unpackCSC,unpackDT,unpackRPC,unpackEmtf,unpackCsctf,unpackBmtf,unpackGtStage2
                                 ,SimL1EmulatorCoreTask,packCaloStage2
                                 ,packGtStage2,rawDataCollector))
SimL1Emulator = cms.Sequence(SimL1EmulatorTask)
