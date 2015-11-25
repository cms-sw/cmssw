#
#  L1TDigiToRaw:  Defines
#
#     L1TDigiToRaw = cms.Sequence(...)
#
# which contains all L1 trigger packers needed for the current era.
#
import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras



#
# Legacy Trigger:
#
if not (eras.stage1L1Trigger.isChosen() or eras.stage2L1Trigger.isChosen()):
    print "L1TDigiToRaw Sequence configured for Run1 (Legacy) trigger. "
    # legacy L1 packages:
    from EventFilter.CSCTFRawToDigi.csctfpacker_cfi import *
    from EventFilter.DTTFRawToDigi.dttfpacker_cfi import *
    from EventFilter.GctRawToDigi.gctDigiToRaw_cfi import *
    from EventFilter.L1GlobalTriggerRawToDigi.l1GtPack_cfi import *
    from EventFilter.L1GlobalTriggerRawToDigi.l1GtEvmPack_cfi import *
    csctfpacker.lctProducer = "simCscTriggerPrimitiveDigis:MPCSORTED"
    csctfpacker.trackProducer = 'simCsctfTrackDigis'
    dttfpacker.DTDigi_Source = 'simDtTriggerPrimitiveDigis'
    dttfpacker.DTTracks_Source = "simDttfDigis:DTTF"
    gctDigiToRaw.rctInputLabel = 'simRctDigis'
    gctDigiToRaw.gctInputLabel = 'simGctDigis'
    l1GtPack.DaqGtInputTag = 'simGtDigis'
    l1GtPack.MuGmtInputTag = 'simGmtDigis'
    l1GtEvmPack.EvmGtInputTag = 'simGtDigis'
    L1TDigiToRaw = cms.Sequence(csctfpacker+dttfpacker+gctDigiToRaw+l1GtPack+l1GtEvmPack)
#
# Stage-1 Trigger
#
if eras.stage1L1Trigger.isChosen() and not eras.stage2L1Trigger.isChosen():
    print "L1TDigiToRaw Sequence configured for Stage-1 (2015) trigger. "    
    # legacy L1 packers, still in use for 2015:
    from EventFilter.CSCTFRawToDigi.csctfpacker_cfi import *
    from EventFilter.DTTFRawToDigi.dttfpacker_cfi import *
    
    from EventFilter.L1GlobalTriggerRawToDigi.l1GtPack_cfi import *
    csctfpacker.lctProducer = "simCscTriggerPrimitiveDigis:MPCSORTED"
    csctfpacker.trackProducer = 'simCsctfTrackDigis'
    dttfpacker.DTDigi_Source = 'simDtTriggerPrimitiveDigis'
    dttfpacker.DTTracks_Source = "simDttfDigis:DTTF"
    l1GtPack.DaqGtInputTag = 'simGtDigis'
    l1GtPack.MuGmtInputTag = 'simGmtDigis'

    # Initially for stage-1, the stage-1 was packed via GCT...
    from EventFilter.GctRawToDigi.gctDigiToRaw_cfi import *
    gctDigiToRaw.gctInputLabel = 'simCaloStage1LegacyFormatDigis'
    gctDigiToRaw.rctInputLabel = 'simRctDigis'
    from EventFilter.L1TRawToDigi.caloStage1Raw_cfi import *
    L1TDigiToRaw = cms.Sequence(csctfpacker+dttfpacker+gctDigiToRaw+l1GtPack+caloStage1Raw)
#   +caloStage1Raw 

#
# Stage-2 Trigger
#
if eras.stage2L1Trigger.isChosen():
    print "L1TDigiToRaw Sequence configured for Stage-2 (2016) trigger. "
    from EventFilter.L1TRawToDigi.caloStage2Raw_cfi import *
    from EventFilter.L1TRawToDigi.gmtStage2Raw_cfi import *
    L1TDigiToRaw = cms.Sequence(caloStage2Raw + gmtStage2Raw)
    # Missing: global trigger, muon TFs, calo layer1, 

