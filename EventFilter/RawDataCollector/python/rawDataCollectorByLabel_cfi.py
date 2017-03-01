import FWCore.ParameterSet.Config as cms

rawDataCollector = cms.EDProducer("RawDataCollectorByLabel",
    verbose = cms.untracked.int32(1),     # 0 = quiet, 1 = collection list, 2 = FED list
    RawCollectionList = cms.VInputTag( cms.InputTag('SiStripDigiToZSRaw'),
                                       cms.InputTag('rawDataCollector'))
)

#
# Make changes if using the Stage 1 trigger
#
from Configuration.Eras.Modifier_stage1L1Trigger_cff import stage1L1Trigger
stage1L1Trigger.toModify( rawDataCollector.RawCollectionList, func = lambda list: list.append(cms.InputTag("l1tDigiToRaw")) )
