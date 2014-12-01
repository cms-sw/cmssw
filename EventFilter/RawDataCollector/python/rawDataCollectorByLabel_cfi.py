import FWCore.ParameterSet.Config as cms

rawDataCollector = cms.EDProducer("RawDataCollectorByLabel",
    verbose = cms.untracked.int32(1),     # 0 = quiet, 1 = collection list, 2 = FED list
    RawCollectionList = cms.VInputTag( cms.InputTag('SiStripDigiToZSRaw'),
                                       cms.InputTag('rawDataCollector'))
)

#
# Make changes for Run 2
#
from Configuration.StandardSequences.Eras import eras
eras.run2.toModify( rawDataCollector.RawCollectionList, func = lambda list: list.append(cms.InputTag("l1tDigiToRaw")) )
