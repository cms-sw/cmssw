import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.DigiToRaw_cff import *


#for _entry in [csctfpacker,dttfpacker,gctDigiToRaw,l1GtPack,l1GtEvmPack,siPixelRawData,SiStripDigiToRaw,castorRawData]:
#    DigiToRaw.remove(_entry)
for _entry in [siPixelRawData,SiStripDigiToRaw,castorRawData]:
    DigiToRaw.remove(_entry)

#for _entry in [cms.InputTag("SiStripDigiToRaw"), cms.InputTag("castorRawData"),cms.InputTag("siPixelRawData"),cms.InputTag("csctfpacker","CSCTFRawData"),cms.InputTag("dttfpacker"),cms.InputTag("gctDigiToRaw"),cms.InputTag("l1GtEvmPack"), cms.InputTag("l1GtPack")]:
#    rawDataCollector.RawCollectionList.remove(_entry)
