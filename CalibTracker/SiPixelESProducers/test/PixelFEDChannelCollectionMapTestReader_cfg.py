import FWCore.ParameterSet.Config as cms

process = cms.Process("ProcessOne")

##
## MessageLogger
##
process.load('FWCore.MessageService.MessageLogger_cfi')   
process.MessageLogger.cerr.enable = False
process.MessageLogger.SiPixelFEDChannelContainer=dict()  
process.MessageLogger.PixelFEDChannelCollectionMapTestReader=dict()  
process.MessageLogger.cout = cms.untracked.PSet(
    enable = cms.untracked.bool(True),
    enableStatistics = cms.untracked.bool(True),
    threshold = cms.untracked.string("INFO"),
    default   = cms.untracked.PSet(limit = cms.untracked.int32(0)),                       
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1),
                                   reportEvery = cms.untracked.int32(1000)
                                   ),                                                      
    SiPixelFEDChannelContainer               = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
    PixelFEDChannelCollectionMapTestReader   = cms.untracked.PSet( limit = cms.untracked.int32(-1))
    )

##
## Empty Source
##
process.source = cms.Source("EmptyIOVSource",
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            lastValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )
##
## Get the payload
##
from CondCore.CondDB.CondDB_cfi import *
CondDBQualityCollection = CondDB.clone(connect = cms.string("sqlite_file:SiPixelStatusScenarios_v1.db"))

process.dbInput = cms.ESSource("PoolDBESSource",
                               CondDBQualityCollection,
                               toGet = cms.VPSet(cms.PSet(record = cms.string('SiPixelStatusScenariosRcd'),
                                                          tag = cms.string('SiPixelFEDChannelContainer_StuckTBM_2018_v1_mc') # choose tag you want
                                                          )
                                                 )
                               )

#from CalibTracker.SiPixelESProducers.PixelFEDChannelCollectionProducer_cfi import *
process.load("CalibTracker.SiPixelESProducers.PixelFEDChannelCollectionProducer_cfi")
#process.SiPixelFEDChannelContainerESProducer = cms.ESProducer("PixelFEDChannelCollectionProducer")


## Retrieve it and check it's available in the ES
##
process.get = cms.EDAnalyzer("EventSetupRecordDataGetter",
                             toGet = cms.VPSet(cms.PSet(record = cms.string('SiPixelFEDChannelContainerESProducerRcd'),
                                                        data = cms.vstring('PixelFEDChannelCollectionMap')
                                                        )
                                               ),
                             verbose = cms.untracked.bool(True)
                             )

##
## Retrieve it and check it's available in the ES
##
# process.get = cms.EDAnalyzer("EventSetupRecordDataGetter",
#                              toGet = cms.VPSet(cms.PSet(record = cms.string('SiPixelStatusScenariosRcd'),
#                                                         data = cms.vstring('SiPixelFEDChannelContainer')
#                                                         )
#                                                ),
#                              verbose = cms.untracked.bool(True)
#                              )

##
## Read it back
##
process.ReadDB = cms.EDAnalyzer("PixelFEDChannelCollectionMapTestReader")
process.ReadDB.printDebug = cms.untracked.bool(True)
process.ReadDB.outputFile = cms.untracked.string('PixelFEDChannelCollectionMap.log')

process.p = cms.Path(process.get+process.ReadDB)
