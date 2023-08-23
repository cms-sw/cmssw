from __future__ import print_function
import FWCore.ParameterSet.Config as cms

## L1REPACK uGT : Re-Emulate L1 uGT and repack into RAW

from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger

(~stage2L1Trigger).toModify(None, lambda x:
    print("# L1T WARN:  L1REPACK:uGT only supports Stage-2 eras for now.\n# L1T WARN:  Use a legacy version of L1REPACK for now."))
stage2L1Trigger.toModify(None, lambda x:
    print("# L1T INFO:  L1REPACK:uGT will unpack uGMT and CaloLayer2 outputs, and re-emulate uGT"))

# First, inputs to uGT:
import EventFilter.L1TRawToDigi.gtStage2Digis_cfi
unpackGtStage2 = EventFilter.L1TRawToDigi.gtStage2Digis_cfi.gtStage2Digis.clone(
    InputLabel = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess()))

import EventFilter.OnlineMetaDataRawToDigi.tcdsRawToDigi_cfi
unpackTcds = EventFilter.OnlineMetaDataRawToDigi.tcdsRawToDigi_cfi.tcdsRawToDigi.clone(
    InputLabel = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess()))

from L1Trigger.Configuration.SimL1Emulator_cff import *

simGtExtFakeStage2Digis.tcdsRecordLabel= cms.InputTag("unpackTcds","tcdsRecord")

simGtStage2Digis.MuonInputTag   = "unpackGtStage2:Muon"
simGtStage2Digis.MuonShowerInputTag = "unpackGtStage2:MuonShower"
simGtStage2Digis.EGammaInputTag = "unpackGtStage2:EGamma"
simGtStage2Digis.TauInputTag    = "unpackGtStage2:Tau"
simGtStage2Digis.JetInputTag    = "unpackGtStage2:Jet"
simGtStage2Digis.EtSumInputTag  = "unpackGtStage2:EtSum"
simGtStage2Digis.ExtInputTag    = "unpackGtStage2" # as in default


# Finally, pack the new L1T output back into RAW
    
# pack simulated uGT
from EventFilter.L1TRawToDigi.gtStage2Raw_cfi import gtStage2Raw as packGtStage2
packGtStage2.MuonInputTag   = "unpackGtStage2:Muon"
packGtStage2.ShowerInputLabel = "unpackGtStage2:MuonShower"
packGtStage2.EGammaInputTag = "unpackGtStage2:EGamma"
packGtStage2.TauInputTag    = "unpackGtStage2:Tau"
packGtStage2.JetInputTag    = "unpackGtStage2:Jet"
packGtStage2.EtSumInputTag  = "unpackGtStage2:EtSum"
packGtStage2.GtInputTag     = "simGtStage2Digis" # as in default
packGtStage2.ExtInputTag    = "unpackGtStage2" # as in default
    

# combine the new L1 RAW with existing RAW for other FEDs
import EventFilter.RawDataCollector.rawDataCollectorByLabel_cfi
rawDataCollector = EventFilter.RawDataCollector.rawDataCollectorByLabel_cfi.rawDataCollector.clone(
    verbose = 0,
    RawCollectionList = [
        'packGtStage2',
        cms.InputTag('rawDataCollector', processName=cms.InputTag.skipCurrentProcess()),
        ]
    )


SimL1EmulatorTask = cms.Task()
stage2L1Trigger.toReplaceWith(SimL1EmulatorTask, cms.Task(unpackGtStage2
                                                          ,unpackTcds
                                                          ,SimL1TechnicalTriggersTask
                                                          ,SimL1TGlobalTask
                                                          ,packGtStage2
                                                          ,rawDataCollector))
SimL1Emulator = cms.Sequence(SimL1EmulatorTask)
