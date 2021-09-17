import FWCore.ParameterSet.Config as cms

process = cms.Process("ProcessOne")

##
## MessageLogger
##
process.load('FWCore.MessageService.MessageLogger_cfi')   
process.MessageLogger.cerr.enable = False
process.MessageLogger.SiPixelBadFEDChannelSimulationSanityChecker=dict()  
process.MessageLogger.SiPixelFEDChannelContainer=dict()
process.MessageLogger.SiPixelQualityProbabilities=dict()    
process.MessageLogger.cout = cms.untracked.PSet(
    enable    = cms.untracked.bool(True),
    enableStatistics = cms.untracked.bool(True),
    threshold = cms.untracked.string("INFO"),
    default   = cms.untracked.PSet(limit = cms.untracked.int32(0)),                       
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1),
                                   reportEvery = cms.untracked.int32(1000)
                                   ),                                                      
    SiPixelBadFEDChannelSimulationSanityChecker  = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
    SiPixelFEDChannelContainer              = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
    SiPixelQualityProbabilities             = cms.untracked.PSet( limit = cms.untracked.int32(-1))
    )
process.MessageLogger.cout.enableStatistics = cms.untracked.bool(True)  

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
## Get the payload(s)
##
from CondCore.CondDB.CondDB_cfi import *
#CondDBQualityCollection = CondDB.clone(connect = cms.string("frontier://FrontierPrep/CMS_CONDITIONS"))
CondDBQualityCollection = CondDB.clone(connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS"))
process.dbInput = cms.ESSource("PoolDBESSource",
                               CondDBQualityCollection,
                               toGet = cms.VPSet(cms.PSet(record = cms.string('SiPixelStatusScenariosRcd'),
                                                          tag = cms.string('SiPixelStatusScenarios_UltraLegacy2018_v0_mc') # choose tag you want
                                                          )
                                                 )
                               )

#CondDBProbabilities = CondDB.clone(connect = cms.string("frontier://FrontierPrep/CMS_CONDITIONS"))
CondDBProbabilities = CondDB.clone(connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS"))
process.dbInput2 = cms.ESSource("PoolDBESSource",
                                CondDBProbabilities,
                                toGet = cms.VPSet(cms.PSet(record = cms.string('SiPixelStatusScenarioProbabilityRcd'),
                                                           tag = cms.string('SiPixelQualityProbabilities_UltraLegacy2018_v0_mc') # choose tag you want
                                                           )
                                                  )
                                )

##
## Retrieve it and check it's available in the ES
##
process.get = cms.EDAnalyzer("EventSetupRecordDataGetter",
                             toGet = cms.VPSet(cms.PSet(record = cms.string('SiPixelStatusScenariosRcd'),
                                                        data = cms.vstring('SiPixelFEDChannelContainer')
                                                        ),
                                               cms.PSet(record = cms.string('SiPixelStatusScenarioProbabilityRcd'),
                                                        data = cms.vstring('SiPixelQualityProbabilities')
                                                        )
                                               ),
                             verbose = cms.untracked.bool(True)
                             )
##
## Read it back
##
process.ReadDB = cms.EDAnalyzer("SiPixelBadFEDChannelSimulationSanityChecker")
process.ReadDB.printDebug = cms.untracked.bool(True)
process.p = cms.Path(process.get+process.ReadDB)
