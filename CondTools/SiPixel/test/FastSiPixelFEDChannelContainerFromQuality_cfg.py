import FWCore.ParameterSet.Config as cms
import os

process = cms.Process("summary")

import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing()
options.register('outputDB',
                 'sqlite_file:SiPixelStatusScenarios_2017StuckTBM.db', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "output conditions DB")
options.register('outputTag',
                 'SiPixelFEDChannelContainer_StuckTBM_2017_v1_mc', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "output conditions tag")
options.register('inputTag',
                 'SiPixelQualityOffline_2017_threshold1percent_stuckTBM', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "input conditions tag")
options.register('firstIOV',
                 1318907147190984, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "first IOV")
options.register('lastIOV',
                 1318907147190984, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "last IOV")
options.parseArguments()

##
## MessageLogger
##
process.load('FWCore.MessageService.MessageLogger_cfi')   
process.MessageLogger.cout.enable = True
process.MessageLogger.FastSiPixelFEDChannelContainerFromQuality=dict()  
process.MessageLogger.SiPixelFEDChannelContainer=dict()  
process.MessageLogger.cout = cms.untracked.PSet(
    threshold = cms.untracked.string("INFO"),
    enableStatistics = cms.untracked.bool(True),
    default   = cms.untracked.PSet(limit = cms.untracked.int32(0)),                       
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1),
                                   reportEvery = cms.untracked.int32(1000)
                                   ),                                                      
    FastSiPixelFEDChannelContainerFromQuality = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
    SiPixelFEDChannelContainer           = cms.untracked.PSet( limit = cms.untracked.int32(-1))
    )
  
##
## Empty Source
##                                      
process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

##
## Output database (in this case local sqlite file)
##
process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = options.outputDB
#process.CondDB.connect = 'sqlite_file:SiPixelStatusScenarios_2017StuckTBM.db'
#process.CondDB.connect = 'sqlite_file:SiPixelStatusScenarios_2017Prompt.db'
#process.CondDB.connect = 'sqlite_file:SiPixelStatusScenarios_2017Other.db'
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDB,
                                          timetype = cms.untracked.string('runnumber'),
                                          toPut = cms.VPSet(cms.PSet(record = cms.string('SiPixelStatusScenariosRcd'),
                                                                     tag = cms.string(options.outputTag)
                                                                     #tag = cms.string('SiPixelFEDChannelContainer_StuckTBM_2017_v1_mc')
                                                                     #tag = cms.string('SiPixelFEDChannelContainer_Prompt_2017_v1_mc')
                                                                     #tag = cms.string('SiPixelFEDChannelContainer_Other_2017_v1_mc')
                                                                     )
                                                            )
                                          )

##
## Configuration of the module
##

print("Processing %s from %s to %s " % (options.inputTag,options.firstIOV,options.lastIOV) )

process.load("CondTools.SiPixel.FastSiPixelFEDChannelContainerFromQuality_cfi")
process.FastSiPixelFEDChannelContainerFromQuality.qualityTagName  = options.inputTag
process.FastSiPixelFEDChannelContainerFromQuality.startIOV = options.firstIOV
process.FastSiPixelFEDChannelContainerFromQuality.endIOV   = options.lastIOV
process.FastSiPixelFEDChannelContainerFromQuality.output   = "summary_StuckTBM_test.txt"

#process.FastSiPixelFEDChannelContainerFromQuality.qualityTagName  = "SiPixelQualityOffline_2017_threshold1percent_stuckTBM"
#process.FastSiPixelFEDChannelContainerFromQuality.startIOV = 1268368267018245
#process.FastSiPixelFEDChannelContainerFromQuality.endIOV   = 1318907147191631
#process.FastSiPixelFEDChannelContainerFromQuality.output   = "summary2017_StuckTBM.txt"

#process.FastSiPixelFEDChannelContainerFromQuality.qualityTagName  = "SiPixelQualityOffline_2017_threshold1percent_prompt"
#process.FastSiPixelFEDChannelContainerFromQuality.startIOV = 1268368267018245
#process.FastSiPixelFEDChannelContainerFromQuality.endIOV   = 1318907147191657
#process.FastSiPixelFEDChannelContainerFromQuality.output   = "summary2017_Prompt.txt"

#process.FastSiPixelFEDChannelContainerFromQuality.qualityTagName  = "SiPixelQualityOffline_2017_threshold1percent_other"
#process.FastSiPixelFEDChannelContainerFromQuality.startIOV = 1268368267018245
#process.FastSiPixelFEDChannelContainerFromQuality.endIOV   = 1318907147191657
#process.FastSiPixelFEDChannelContainerFromQuality.output   = "summary2017_Other.txt"

process.p = cms.Path(process.FastSiPixelFEDChannelContainerFromQuality)
