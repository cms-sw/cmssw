import FWCore.ParameterSet.Config as cms
process = cms.Process("ProcessOne")

##
## MessageLogger
##
process.load('FWCore.MessageService.MessageLogger_cfi')   
process.MessageLogger.categories.append("SiPixelFEDChannelContainerWriteFromASCII")  
process.MessageLogger.categories.append("SiPixelFEDChannelContainer")  
process.MessageLogger.destinations = cms.untracked.vstring("cout")
process.MessageLogger.cout = cms.untracked.PSet(
    threshold = cms.untracked.string("INFO"),
    default   = cms.untracked.PSet(limit = cms.untracked.int32(0)),                       
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1),
                                   reportEvery = cms.untracked.int32(100)
                                   ),                                                      
    SiPixelFEDChannelContainerWriteFromASCII = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
    SiPixelFEDChannelContainer           = cms.untracked.PSet( limit = cms.untracked.int32(-1))
    )
process.MessageLogger.statistics.append('cout')  

##
## Empty source
##
process.source = cms.Source("EmptySource",
                            firstRun = cms.untracked.uint32(1),
                            numberEventsInRun    = cms.untracked.uint32(1),
                            firstLuminosityBlock = cms.untracked.uint32(1),
                            numberEventsInLuminosityBlock = cms.untracked.uint32(1),
                            )

#process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(25000000))
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1))

##
## Database output service
##
process.load("CondCore.CondDB.CondDB_cfi")

##
## Output database (in this case local sqlite file)
##
process.CondDB.connect = 'sqlite_file:SiPixelStatusScenarios_UltraLegacy2018_v0_mc.db'
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDB,
                                          timetype = cms.untracked.string('runnumber'),
                                          toPut = cms.VPSet(cms.PSet(record = cms.string('SiPixelStatusScenariosRcd'),
                                                                     tag = cms.string('SiPixelStatusScenarios_UltraLegacy2018_v0_mc')
                                                                     )
                                                            )
                                          )

process.WriteInDB = cms.EDAnalyzer("SiPixelFEDChannelContainerWriteFromASCII",
                                   record     = cms.string('SiPixelStatusScenariosRcd'),
                                   snapshots  = cms.string('scenarios_2018_-1.txt'),
                                   printDebug = cms.untracked.bool(False),
                                   addDefault = cms.untracked.bool(False)
                                   )

process.p = cms.Path(process.WriteInDB)
