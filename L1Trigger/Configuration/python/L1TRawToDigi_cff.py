#
#  L1TRawToDigi:  Defines
#
#     L1TRawToDigi = cms.Sequence(...)
#
# which contains all packers needed for the current era.
#

#
# There are 3 Hacks below which need to be fixed before this can be used generally...
#

import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

#
# Legacy Trigger:
#
if not (eras.stage1L1Trigger.isChosen() or eras.stage2L1Trigger.isChosen()):
    print "L1TRawToDigi Sequence configured for Run1 (Legacy) trigger. "
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
    L1TRawToDigi = cms.Sequence(csctfDigis+dttfDigis+gctDigis+gtDigis+gtEvmDigis)
#
# Stage-1 Trigger
#
if eras.stage1L1Trigger.isChosen() and not eras.stage2L1Trigger.isChosen():
    print "L1TRawToDigi Sequence configured for Stage-1 (2015) trigger. "    
    import EventFilter.CSCTFRawToDigi.csctfunpacker_cfi
    csctfDigis = EventFilter.CSCTFRawToDigi.csctfunpacker_cfi.csctfunpacker.clone()
    import EventFilter.DTTFRawToDigi.dttfunpacker_cfi
    dttfDigis = EventFilter.DTTFRawToDigi.dttfunpacker_cfi.dttfunpacker.clone()
    import EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi
    gtDigis = EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi.l1GtUnpack.clone()
    from EventFilter.L1TRawToDigi.caloStage1Digis_cfi import *
    # this adds the physical ET to unpacked data
    import L1Trigger.L1TCalorimeter.simCaloStage1FinalDigis_cfi
    caloStage1FinalDigis = L1Trigger.L1TCalorimeter.simCaloStage1FinalDigis_cfi.simCaloStage1FinalDigis.clone()    
    caloStage1FinalDigis.InputCollection = cms.InputTag("caloStage1Digis")
    caloStage1FinalDigis.InputRlxTauCollection = cms.InputTag("caloStage1Digis:rlxTaus")
    caloStage1FinalDigis.InputIsoTauCollection = cms.InputTag("caloStage1Digis:isoTaus")
    caloStage1FinalDigis.InputPreGtJetCollection = cms.InputTag("caloStage1Digis") # not sure whether this is right...
    caloStage1FinalDigis.InputHFSumsCollection = cms.InputTag("caloStage1Digis:HFRingSums")
    caloStage1FinalDigis.InputHFCountsCollection = cms.InputTag("caloStage1Digis:HFBitCounts")
    csctfDigis.producer = 'rawDataCollector'
    dttfDigis.DTTF_FED_Source = 'rawDataCollector'
    gtDigis.DaqGtInputTag = 'rawDataCollector'
    L1TRawToDigi = cms.Sequence(csctfDigis+dttfDigis+gtDigis+caloStage1Digis+caloStage1FinalDigis)    

#
# Stage-2 Trigger
#
if eras.stage2L1Trigger.isChosen():
    print "L1TRawToDigi Sequence configured for Stage-2 (2016) trigger. "
    from EventFilter.L1TRawToDigi.caloStage2Digis_cfi import *
    from EventFilter.L1TRawToDigi.gmtStage2Digis_cfi import *
    from EventFilter.L1TRawToDigi.gtStage2Digis_cfi import *
    L1TRawToDigi = cms.Sequence(caloStage2Digis + gmtStage2Digis + gtStage2Digis)

