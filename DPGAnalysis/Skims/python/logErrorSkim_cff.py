from FWCore.Modules.logErrorFilter_cfi import *

stableBeam = cms.EDFilter("HLTBeamModeFilter",
                          L1GtEvmReadoutRecordTag = cms.InputTag("gtEvmDigis"),
                          AllowedBeamMode = cms.vuint32(11),
                          saveTags = cms.bool(False)
                          )
from L1Trigger.Configuration.L1TRawToDigi_cff import L1TRawToDigi
if not "gtEvmDigis" in L1TRawToDigi.moduleNames():
    import EventFilter.L1GlobalTriggerRawToDigi.l1GtEvmUnpack_cfi
    gtEvmDigis = EventFilter.L1GlobalTriggerRawToDigi.l1GtEvmUnpack_cfi.l1GtEvmUnpack.clone()
    gtEvmDigis.EvmGtInputTag = 'rawDataCollector'
else:
    from L1Trigger.Configuration.L1TRawToDigi_cff import gtEvmDigis

logerrorseq=cms.Sequence(gtEvmDigis+stableBeam+logErrorSkimFilter)
logerrormonitorseq=cms.Sequence(gtEvmDigis+stableBeam+logErrorFilter)
