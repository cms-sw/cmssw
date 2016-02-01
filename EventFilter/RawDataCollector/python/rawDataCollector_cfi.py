import FWCore.ParameterSet.Config as cms

rawDataCollector = cms.EDProducer("RawDataCollectorByLabel",
    verbose = cms.untracked.int32(0),     # 0 = quiet, 1 = collection list, 2 = FED list
    RawCollectionList = cms.VInputTag( cms.InputTag('SiStripDigiToRaw'),
                                       cms.InputTag('castorRawData'),
                                       cms.InputTag('cscpacker', 'CSCRawData'),
                                       cms.InputTag('csctfpacker', 'CSCTFRawData'),
                                       cms.InputTag('dtpacker'),
                                       cms.InputTag('dttfpacker'),
                                       cms.InputTag('ecalPacker'),
                                       cms.InputTag('esDigiToRaw'),
                                       cms.InputTag('gctDigiToRaw'),
                                       cms.InputTag('hcalRawData'),
                                       cms.InputTag('l1GtEvmPack'),
                                       cms.InputTag('l1GtPack'),
                                       cms.InputTag('rpcpacker'),
                                       cms.InputTag('siPixelRawData')
    ),
)

#
# Make changes if using the Stage 1 trigger
#
from Configuration.StandardSequences.Eras import eras
eras.stage1L1Trigger.toModify( rawDataCollector.RawCollectionList, func = lambda list: list.append(cms.InputTag("l1tDigiToRaw")) )
