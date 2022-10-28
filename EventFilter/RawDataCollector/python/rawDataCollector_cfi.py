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
                                       cms.InputTag('hcalRawDataVME'),
                                       cms.InputTag('l1GtEvmPack'),
                                       cms.InputTag('l1GtPack'),
                                       cms.InputTag('rpcpacker'),
                                       cms.InputTag('siPixelRawData')
    ),
)

#
# Make changes if using the Stage 1 trigger
#
from Configuration.Eras.Modifier_stage1L1Trigger_cff import stage1L1Trigger
stage1L1Trigger.toModify( rawDataCollector.RawCollectionList, func = lambda list: list.append(cms.InputTag("l1tDigiToRaw")) )
from Configuration.Eras.Modifier_run2_HCAL_2017_cff import run2_HCAL_2017
run2_HCAL_2017.toModify( rawDataCollector.RawCollectionList, func = lambda list: list.append(cms.InputTag("hcalRawDatauHTR")) )


# GEM settings
from Configuration.Eras.Modifier_run2_GEM_2017_cff import run2_GEM_2017
run2_GEM_2017.toModify( rawDataCollector.RawCollectionList, func = lambda list: list.append(cms.InputTag("gemPacker")) )

from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
run3_GEM.toModify( rawDataCollector.RawCollectionList, func = lambda list: list.append(cms.InputTag("gemPacker")) )

# For Run2 it is needed to include the general ctpps era ctpps_2016
from Configuration.Eras.Modifier_ctpps_2022_cff import ctpps_2022
ctpps_2022.toModify(rawDataCollector.RawCollectionList, func = lambda  list: list.extend([cms.InputTag("ctppsTotemRawData"),cms.InputTag("ctppsPixelRawData")]) )

# Phase-2 Tracker
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify( rawDataCollector.RawCollectionList, func = lambda list: list.remove(cms.InputTag("SiStripDigiToRaw")) )
