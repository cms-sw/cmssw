from __future__ import print_function
import FWCore.ParameterSet.Config as cms

## L1REPACK FullMC : Re-Emulate all of L1 and repack into RAW

from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM

(~stage2L1Trigger).toModify(None, lambda x:
    print("# L1T WARN:  L1REPACK:FullMC (intended for MC events with RAW eventcontent) only supports Stage-2 eras for now.\n# L1T WARN:  Use a legacy version of L1REPACK for now."))
stage2L1Trigger.toModify(None, lambda x:
    print("# L1T INFO:  L1REPACK:FullMC will unpack Calorimetry and Muon L1T inputs, re-emulate L1T (Stage-2), and pack uGT, uGMT, and Calo Stage-2 output."))

# First, Unpack all inputs to L1:

import EventFilter.RPCRawToDigi.rpcUnpacker_cfi
unpackRPC = EventFilter.RPCRawToDigi.rpcUnpacker_cfi.rpcunpacker.clone(
    InputLabel = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess()))

import EventFilter.DTRawToDigi.dtunpacker_cfi
unpackDT = EventFilter.DTRawToDigi.dtunpacker_cfi.muonDTDigis.clone(
    inputLabel = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess()))

import EventFilter.CSCRawToDigi.cscUnpacker_cfi
unpackCSC = EventFilter.CSCRawToDigi.cscUnpacker_cfi.muonCSCDigis.clone(
    InputObjects = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess()))

import EventFilter.GEMRawToDigi.muonGEMDigis_cfi
unpackGEM = EventFilter.GEMRawToDigi.muonGEMDigis_cfi.muonGEMDigis.clone(
    InputLabel = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess()))

import EventFilter.EcalRawToDigi.EcalUnpackerData_cfi
unpackEcal = EventFilter.EcalRawToDigi.EcalUnpackerData_cfi.ecalEBunpacker.clone(
    InputLabel = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess()))

import EventFilter.HcalRawToDigi.HcalRawToDigi_cfi
unpackHcal = EventFilter.HcalRawToDigi.HcalRawToDigi_cfi.hcalDigis.clone(
    InputLabel = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess()))

# Second, Re-Emulate the entire L1T
#
# Legacy trigger primitive emulations still running in 2016 trigger:
#
from SimCalorimetry.Configuration.SimCalorimetry_cff import *

# Ecal TPs
# cannot simulate EcalTPs, don't have EcalUnsuppressedDigis in RAW
#     simEcalTriggerPrimitiveDigis.Label = 'unpackEcal'
# further downstream, use unpacked EcalTPs

# Hcal TPs
simHcalTriggerPrimitiveDigis.inputLabel = [
    'unpackHcal',
    'unpackHcal'
]
simHcalTriggerPrimitiveDigis.inputUpgradeLabel = [
    'unpackHcal',     # upgrade HBHE
    'unpackHcal'      # upgrade HF
]

from L1Trigger.Configuration.SimL1Emulator_cff import *
# DT TPs
simDtTriggerPrimitiveDigis.digiTag                    = 'unpackDT'
# CSC TPs
simCscTriggerPrimitiveDigis.CSCComparatorDigiProducer = 'unpackCSC:MuonCSCComparatorDigi'
simCscTriggerPrimitiveDigis.CSCWireDigiProducer       = 'unpackCSC:MuonCSCWireDigi'
# GEM
(stage2L1Trigger & run3_GEM).toModify(simMuonGEMPadDigis, InputCollection = 'unpackGEM')

# TWIN-MUX
simTwinMuxDigis.RPC_Source         = 'unpackRPC'
simTwinMuxDigis.DTDigi_Source      = "simDtTriggerPrimitiveDigis"
simTwinMuxDigis.DTThetaDigi_Source = "simDtTriggerPrimitiveDigis"

# BMTF
simBmtfDigis.DTDigi_Source       = "simTwinMuxDigis"
simBmtfDigis.DTDigi_Theta_Source = "simDtTriggerPrimitiveDigis"

# OMTF
simOmtfDigis.srcRPC              = 'unpackRPC'
simOmtfDigis.srcDTPh             = "simDtTriggerPrimitiveDigis"
simOmtfDigis.srcDTTh             = "simDtTriggerPrimitiveDigis"
simOmtfDigis.srcCSC              = 'simCscTriggerPrimitiveDigis:MPCSORTED'

# EMTF
simEmtfDigis.CSCInput            = 'simCscTriggerPrimitiveDigis:MPCSORTED'
simEmtfDigis.RPCInput            = 'unpackRPC'

# CALO Layer1
simCaloStage2Layer1Digis.ecalToken = 'unpackEcal:EcalTriggerPrimitives'
simCaloStage2Layer1Digis.hcalToken = 'simHcalTriggerPrimitiveDigis'

# Finally, pack the new L1T output back into RAW
from EventFilter.L1TRawToDigi.caloStage2Raw_cfi import caloStage2Raw as packCaloStage2
from EventFilter.L1TRawToDigi.gmtStage2Raw_cfi import gmtStage2Raw as packGmtStage2
from EventFilter.L1TRawToDigi.gtStage2Raw_cfi import gtStage2Raw as packGtStage2

# combine the new L1 RAW with existing RAW for other FEDs
import EventFilter.RawDataCollector.rawDataCollectorByLabel_cfi
rawDataCollector = EventFilter.RawDataCollector.rawDataCollectorByLabel_cfi.rawDataCollector.clone(
    verbose = 0,
        RawCollectionList = [
            'packCaloStage2',
            'packGmtStage2',
            'packGtStage2',
            cms.InputTag('rawDataCollector', processName=cms.InputTag.skipCurrentProcess()),
        ]
    )

SimL1EmulatorTask = cms.Task()
stage2L1Trigger.toReplaceWith(SimL1EmulatorTask, cms.Task(unpackRPC
                                                          , unpackDT
                                                          , unpackCSC
                                                          , unpackEcal
                                                          , unpackHcal
                                                          #, simEcalTriggerPrimitiveDigis
                                                          , simHcalTriggerPrimitiveDigis
                                                          , SimL1EmulatorCoreTask
                                                          , packCaloStage2
                                                          , packGmtStage2
                                                          , packGtStage2
                                                          , rawDataCollector))

_SimL1EmulatorTaskWithGEM = SimL1EmulatorTask.copy()
_SimL1EmulatorTaskWithGEM.add(unpackGEM)
(stage2L1Trigger & run3_GEM).toReplaceWith(SimL1EmulatorTask, _SimL1EmulatorTaskWithGEM)

SimL1Emulator = cms.Sequence(SimL1EmulatorTask)
