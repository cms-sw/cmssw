import FWCore.ParameterSet.Config as cms

process = cms.Process("ProcessOne")

##
## MessageLogger
##
process.load('FWCore.MessageService.MessageLogger_cfi')   
process.MessageLogger.cerr.enable = False
process.MessageLogger.SiPixelQualityProbabilitiesTestReader=dict()  
process.MessageLogger.SiPixelQualityProbabilities=dict()  
process.MessageLogger.cout = cms.untracked.PSet(
    enable    = cms.untracked.bool(True),
    enableStatistics = cms.untracked.bool(True),
    threshold = cms.untracked.string("INFO"),
    default   = cms.untracked.PSet(limit = cms.untracked.int32(0)),                       
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1),
                                   reportEvery = cms.untracked.int32(1000)
                                   ),                                                      
    SiPixelQualityProbabilitiesTestReader = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
    SiPixelQualityProbabilities           = cms.untracked.PSet( limit = cms.untracked.int32(-1))
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
CondDBQualityProbabilities = CondDB.clone(connect = cms.string("sqlite_file:SiPixelStatusScenarioProbabilities.db"))

process.dbInput = cms.ESSource("PoolDBESSource",
                               CondDBQualityProbabilities,
                               toGet = cms.VPSet(cms.PSet(record = cms.string('SiPixelStatusScenarioProbabilityRcd'),
                                                          tag = cms.string('SiPixelQualityProbabilities_v0_mc') # choose tag you want
                                                          )
                                                 )
                               )
##
## Retrieve it and check it's available in the ES
##
process.get = cms.EDAnalyzer("EventSetupRecordDataGetter",
                             toGet = cms.VPSet(cms.PSet(record = cms.string('SiPixelStatusScenarioProbabilityRcd'),
                                                        data = cms.vstring('SiPixelQualityProbabilities')
                                                        )
                                               ),
                             verbose = cms.untracked.bool(True)
                             )
##
## Read it back
##
process.ReadDB = cms.EDAnalyzer("SiPixelQualityProbabilitiesTestReader")
process.ReadDB.printDebug = cms.untracked.bool(True)
process.ReadDB.outputFile = cms.untracked.string('SiPixelQualityProbabilities.log')

process.p = cms.Path(process.get+process.ReadDB)
