#
#  L1TRawToDigi:  Defines
#
#     L1TRawToDigi = cms.Sequence(...)
#
# which contains all packers needed for the current era.
#

import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras


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
    
# still debugging this approach... it seems there are problems with cloning in sub-routines, so moving these out for now.
import L1Trigger.L1TCalorimeter.simCaloStage1FinalDigis_cfi
caloStage1FinalDigis = L1Trigger.L1TCalorimeter.simCaloStage1FinalDigis_cfi.simCaloStage1FinalDigis.clone()    
import L1Trigger.L1TCalorimeter.simCaloStage1LegacyFormatDigis_cfi
caloStage1LegacyFormatDigis = L1Trigger.L1TCalorimeter.simCaloStage1LegacyFormatDigis_cfi.simCaloStage1LegacyFormatDigis.clone();

def unpack_stage1():
    global csctfDigis, dttfDigis, gtDigis,caloStage1Digis,caloStage1FinalDigis
    global L1TRawToDigi_Stage1    
    import EventFilter.CSCTFRawToDigi.csctfunpacker_cfi
    csctfDigis = EventFilter.CSCTFRawToDigi.csctfunpacker_cfi.csctfunpacker.clone()
    import EventFilter.DTTFRawToDigi.dttfunpacker_cfi
    dttfDigis = EventFilter.DTTFRawToDigi.dttfunpacker_cfi.dttfunpacker.clone()
    import EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi
    gtDigis = EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi.l1GtUnpack.clone()
    from EventFilter.L1TRawToDigi.caloStage1Digis_cfi import caloStage1Digis
    # this adds the physical ET to unpacked data
    caloStage1FinalDigis.InputCollection = cms.InputTag("caloStage1Digis")
    caloStage1FinalDigis.InputRlxTauCollection = cms.InputTag("caloStage1Digis:rlxTaus")
    caloStage1FinalDigis.InputIsoTauCollection = cms.InputTag("caloStage1Digis:isoTaus")
    caloStage1FinalDigis.InputPreGtJetCollection = cms.InputTag("caloStage1Digis") # not sure whether this is right...
    caloStage1FinalDigis.InputHFSumsCollection = cms.InputTag("caloStage1Digis:HFRingSums")
    caloStage1FinalDigis.InputHFCountsCollection = cms.InputTag("caloStage1Digis:HFBitCounts")
    caloStage1LegacyFormatDigis.InputCollection = cms.InputTag("caloStage1FinalDigis")
    caloStage1LegacyFormatDigis.InputRlxTauCollection = cms.InputTag("caloStage1Digis:rlxTaus")
    caloStage1LegacyFormatDigis.InputIsoTauCollection = cms.InputTag("caloStage1Digis:isoTaus")
    caloStage1LegacyFormatDigis.InputHFSumsCollection = cms.InputTag("caloStage1Digis:HFRingSums")
    caloStage1LegacyFormatDigis.InputHFCountsCollection = cms.InputTag("caloStage1Digis:HFBitCounts")
    csctfDigis.producer = 'rawDataCollector'
    dttfDigis.DTTF_FED_Source = 'rawDataCollector'
    gtDigis.DaqGtInputTag = 'rawDataCollector'
    L1TRawToDigi_Stage1 = cms.Sequence(csctfDigis+dttfDigis+gtDigis+caloStage1Digis+caloStage1FinalDigis+caloStage1LegacyFormatDigis)    

def unpack_stage2():
    global L1TRawToDigi_Stage2
    global caloStage2Digis, gmtStage2Digis, gtStage2Digis,L1TRawToDigi_Stage2    
    from EventFilter.L1TRawToDigi.caloStage2Digis_cfi import caloStage2Digis
    from EventFilter.L1TRawToDigi.gmtStage2Digis_cfi import gmtStage2Digis
    from EventFilter.L1TRawToDigi.gtStage2Digis_cfi import gtStage2Digis
    L1TRawToDigi_Stage2 = cms.Sequence(caloStage2Digis + gmtStage2Digis + gtStage2Digis)
    
#
# Legacy Trigger:
#
if not (eras.stage1L1Trigger.isChosen() or eras.stage2L1Trigger.isChosen()):
    print "L1TRawToDigi Sequence configured for Run1 (Legacy) trigger. "
    unpack_legacy()
    L1TRawToDigi = cms.Sequence(L1TRawToDigi_Legacy);

#
# Stage-1 Trigger
#
if eras.stage1L1Trigger.isChosen() and not eras.stage2L1Trigger.isChosen():
    print "L1TRawToDigi Sequence configured for Stage-1 (2015) trigger. "    
    unpack_stage1()
    L1TRawToDigi = cms.Sequence(L1TRawToDigi_Stage1);

#
# Stage-2 Trigger:  fow now, unpack Stage 1 and Stage 2 (in case both available)
#
if eras.stage2L1Trigger.isChosen():
    print "L1TRawToDigi Sequence configured for Stage-2 (2016) trigger. "    
    unpack_stage1()
    unpack_stage2()
    L1TRawToDigi = cms.Sequence(L1TRawToDigi_Stage1+L1TRawToDigi_Stage2);
