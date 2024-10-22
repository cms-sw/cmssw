import FWCore.ParameterSet.Config as cms

process = cms.Process("ProcessOne")

import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing()
options.register('conditionsDB',
                 'sqlite_file:SiPixelStatusScenarios_2017StuckTBM.db', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "input conditions DB")
options.register('inputTag',
                 'SiPixelFEDChannelContainer_StuckTBM_2017_v1_mc', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "input conditions tag")
options.parseArguments()

##
## MessageLogger
##
process.load('FWCore.MessageService.MessageLogger_cfi')   
process.MessageLogger.cerr.enable = False
process.MessageLogger.SiPixelFEDChannelContainerTestReader=dict()  
process.MessageLogger.SiPixelFEDChannelContainer=dict()  
process.MessageLogger.cout = cms.untracked.PSet(
    enable    = cms.untracked.bool(True),
    enableStatistics = cms.untracked.bool(True),
    threshold = cms.untracked.string("INFO"),
    default   = cms.untracked.PSet(limit = cms.untracked.int32(0)),                       
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1),
                                   reportEvery = cms.untracked.int32(1000)
                                   ),                                                      
    SiPixelFEDChannelContainerTestReader = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
    SiPixelFEDChannelContainer           = cms.untracked.PSet( limit = cms.untracked.int32(-1))
    )

##
## Empty Source
##
process.source = cms.Source("EmptyIOVSource",
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(315708),
                            lastValue = cms.uint64(315708),
                            interval = cms.uint64(1)
                            )
##
## Get the payload
##
from CondCore.CondDB.CondDB_cfi import *
CondDBQualityCollection = CondDB.clone(connect = cms.string(options.conditionsDB))

process.dbInput = cms.ESSource("PoolDBESSource",
                               CondDBQualityCollection,
                               toGet = cms.VPSet(cms.PSet(record = cms.string('SiPixelStatusScenariosRcd'),
                                                          tag = cms.string(options.inputTag) # choose tag you want
                                                          )
                                                 )
                               )
##
## Retrieve it and check it's available in the ES
##
process.get = cms.EDAnalyzer("EventSetupRecordDataGetter",
                             toGet = cms.VPSet(cms.PSet(record = cms.string('SiPixelStatusScenariosRcd'),
                                                        data = cms.vstring('SiPixelFEDChannelContainer')
                                                        )
                                               ),
                             verbose = cms.untracked.bool(True)
                             )
##
## Read it back
##
process.ReadDB = cms.EDAnalyzer("SiPixelFEDChannelContainerTestReader")
process.ReadDB.printDebug = cms.untracked.bool(True)
process.ReadDB.outputFile = cms.untracked.string('SiPixelFEDChannelContainer.log')

process.p = cms.Path(process.get+process.ReadDB)
