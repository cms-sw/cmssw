#
#  L1TRawToDigi:  Defines
#
#     L1TRawToDigi = cms.Sequence(...)
#
# which contains all packers needed for the current era.
#

import FWCore.ParameterSet.Config as cms


def unpack_legacy():
    global L1TRawToDigi_Legacy
    global csctfDigis, dttfDigis, gctDigis, gtDigis, gtEvmDigis
    import EventFilter.CSCTFRawToDigi.csctfunpacker_cfi
    csctfDigis = EventFilter.CSCTFRawToDigi.csctfunpacker_cfi.csctfunpacker.clone()
    import EventFilter.DTTFRawToDigi.dttfunpacker_cfi
    dttfDigis = EventFilter.DTTFRawToDigi.dttfunpacker_cfi.dttfunpacker.clone()
    import EventFilter.GctRawToDigi.l1GctHwDigis_cfi
    gctDigis = EventFilter.GctRawToDigi.l1GctHwDigis_cfi.l1GctHwDigis.clone()
    import EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi
    gtDigis = EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi.l1GtUnpack.clone()
    import EventFilter.L1GlobalTriggerRawToDigi.l1GtEvmUnpack_cfi
    gtEvmDigis = EventFilter.L1GlobalTriggerRawToDigi.l1GtEvmUnpack_cfi.l1GtEvmUnpack.clone()
    #
    csctfDigis.producer = 'rawDataCollector'
    dttfDigis.DTTF_FED_Source = 'rawDataCollector'
    gctDigis.inputLabel = 'rawDataCollector'
    gtDigis.DaqGtInputTag = 'rawDataCollector'
    gtEvmDigis.EvmGtInputTag = 'rawDataCollector'
    L1TRawToDigi_Legacy = cms.Sequence(csctfDigis+dttfDigis+gctDigis+gtDigis+gtEvmDigis)
    

def unpack_stage1():
    global csctfDigis, dttfDigis, gtDigis,caloStage1Digis,caloStage1FinalDigis,gctDigis
    global caloStage1LegacyFormatDigis
    global L1TRawToDigi_Stage1    
    import EventFilter.CSCTFRawToDigi.csctfunpacker_cfi
    csctfDigis = EventFilter.CSCTFRawToDigi.csctfunpacker_cfi.csctfunpacker.clone()
    import EventFilter.DTTFRawToDigi.dttfunpacker_cfi
    dttfDigis = EventFilter.DTTFRawToDigi.dttfunpacker_cfi.dttfunpacker.clone()
    import EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi
    gtDigis = EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi.l1GtUnpack.clone()
    from EventFilter.L1TRawToDigi.caloStage1Digis_cfi import caloStage1Digis
    # this adds the physical ET to unpacked data
    from L1Trigger.L1TCalorimeter.caloStage1LegacyFormatDigis_cfi import caloStage1LegacyFormatDigis
    from L1Trigger.L1TCalorimeter.caloStage1FinalDigis_cfi import caloStage1FinalDigis
    csctfDigis.producer = 'rawDataCollector'
    dttfDigis.DTTF_FED_Source = 'rawDataCollector'
    gtDigis.DaqGtInputTag = 'rawDataCollector'
    # unpack GCT digis too, so DQM offline doesn't crash:
    import EventFilter.GctRawToDigi.l1GctHwDigis_cfi
    gctDigis = EventFilter.GctRawToDigi.l1GctHwDigis_cfi.l1GctHwDigis.clone()
    gctDigis.inputLabel = 'rawDataCollector'
    L1TRawToDigi_Stage1 = cms.Sequence(csctfDigis+dttfDigis+gtDigis+caloStage1Digis+caloStage1FinalDigis+caloStage1LegacyFormatDigis+gctDigis)    

def unpack_stage2():
    global L1TRawToDigi_Stage2
    global bmtfDigis, emtfStage2Digis, caloStage2Digis, gmtStage2Digis, gtStage2Digis,L1TRawToDigi_Stage2    
    from EventFilter.L1TRawToDigi.bmtfDigis_cfi import bmtfDigis 
    from EventFilter.L1TRawToDigi.emtfStage2Digis_cfi import emtfStage2Digis
    from EventFilter.L1TRawToDigi.caloStage2Digis_cfi import caloStage2Digis
    from EventFilter.L1TRawToDigi.gmtStage2Digis_cfi import gmtStage2Digis
    from EventFilter.L1TRawToDigi.gtStage2Digis_cfi import gtStage2Digis
    L1TRawToDigi_Stage2 = cms.Sequence(bmtfDigis + emtfStage2Digis + caloStage2Digis + gmtStage2Digis + gtStage2Digis)
    
#
# Legacy Trigger:
#
from Configuration.Eras.Modifier_stage1L1Trigger_cff import stage1L1Trigger
from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
if not (stage1L1Trigger.isChosen() or stage2L1Trigger.isChosen() or phase2_common.isChosen()):
    print "L1TRawToDigi Sequence configured for Run1 (Legacy) trigger. "
    unpack_legacy()
    L1TRawToDigi = cms.Sequence(L1TRawToDigi_Legacy);

#
# Stage-1 Trigger
#
if stage1L1Trigger.isChosen() and not (stage2L1Trigger.isChosen() or phase2_common.isChosen()):
    print "L1TRawToDigi Sequence configured for Stage-1 (2015) trigger. "    
    unpack_stage1()
    L1TRawToDigi = cms.Sequence(L1TRawToDigi_Stage1)

#
# Stage-2 Trigger:  fow now, unpack Stage 1 and Stage 2 (in case both available)
#
if (stage2L1Trigger.isChosen() or phase2_common.isChosen()):
    print "L1TRawToDigi Sequence configured for Stage-2 (2016) trigger. "    
    unpack_stage1()
    unpack_stage2()
    L1TRawToDigi = cms.Sequence(L1TRawToDigi_Stage1+L1TRawToDigi_Stage2)
    # we only warn if it is stage-2 era and it is an essential, always present, stage-2 payload:
    caloStage2Digis.MinFeds = cms.uint32(1)
    gmtStage2Digis.MinFeds = cms.uint32(1)
    gtStage2Digis.MinFeds = cms.uint32(1)
    
