import FWCore.ParameterSet.Config as cms
process = cms.Process("ProcessOne")

##
## MessageLogger
##
process.load('FWCore.MessageService.MessageLogger_cfi')   
process.MessageLogger.cerr.enable = False
process.MessageLogger.SiPixelFEDChannelContainerFromQualityConverter=dict()  
process.MessageLogger.SiPixelFEDChannelContainer=dict()  
process.MessageLogger.cout = cms.untracked.PSet(
    enable    = cms.untracked.bool(True),
    enableStatistics = cms.untracked.bool(True),
    threshold = cms.untracked.string("INFO"),
    default   = cms.untracked.PSet(limit = cms.untracked.int32(0)),                       
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1),
                                   reportEvery = cms.untracked.int32(1000)
                                   ),                                                      
    SiPixelFEDChannelContainerFromQualityConverter = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
    SiPixelFEDChannelContainer           = cms.untracked.PSet( limit = cms.untracked.int32(-1))
    )

##
## Empty source
##
process.source = cms.Source("EmptySource",
                            firstRun = cms.untracked.uint32(315704),
                            numberEventsInRun    = cms.untracked.uint32(2000),
                            firstLuminosityBlock = cms.untracked.uint32(1),
                            numberEventsInLuminosityBlock = cms.untracked.uint32(1),
                            )

#process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(25000000))
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10000))

##
## Database output service
##
process.load("CondCore.CondDB.CondDB_cfi")

# DB input service: 
process.CondDB.connect = "frontier://FrontierProd/CMS_CONDITIONS"
process.dbInput = cms.ESSource("PoolDBESSource",
                               process.CondDB,
                               toGet = cms.VPSet(cms.PSet(record = cms.string("SiPixelQualityFromDbRcd"),
                                                          #tag = cms.string("SiPixelQuality_byPCL_stuckTBM_v1")
                                                          tag = cms.string("SiPixelQuality_byPCL_other_v1")
                                                          ),
                                                 cms.PSet(record = cms.string("SiPixelFedCablingMapRcd"),
                                                          tag = cms.string("SiPixelFedCablingMap_phase1_v7")
                                                          )
                                                 )
                               )
##
## Output database (in this case local sqlite file)
##
process.CondDB.connect = 'sqlite_file:SiPixelStatusScenarios_v2.db'
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDB,
                                          timetype = cms.untracked.string('runnumber'),
                                          toPut = cms.VPSet(cms.PSet(record = cms.string('SiPixelStatusScenariosRcd'),
                                                                     tag = cms.string('SiPixelFEDChannelContainer_StuckTBM_2018_v1_mc')
                                                                     )
                                                            )
                                          )

process.WriteInDB = cms.EDAnalyzer("SiPixelFEDChannelContainerFromQualityConverter",
                                   record = cms.string('SiPixelStatusScenariosRcd'),
                                   removeEmptyPayloads = cms.untracked.bool(True)
                                   )

process.p = cms.Path(process.WriteInDB)
