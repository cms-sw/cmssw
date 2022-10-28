import FWCore.ParameterSet.Config as cms
process = cms.Process("ProcessOne")

##
## MessageLogger
##
process.load('FWCore.MessageService.MessageLogger_cfi')   
process.MessageLogger.cerr.enable = False
process.MessageLogger.SiPixelQualityProbabilitiesWriteFromASCII=dict()  
process.MessageLogger.SiPixelQualityProbabilities=dict()  
process.MessageLogger.cout = cms.untracked.PSet(
    enable = cms.untracked.bool(True),
    enableStatistics = cms.untracked.bool(True),
    threshold = cms.untracked.string("INFO"),
    default   = cms.untracked.PSet(limit = cms.untracked.int32(0)),                       
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1),
                                   reportEvery = cms.untracked.int32(1000)
                                   ),                                                      
    SiPixelQualityProbabilitiesWriteFromASCII = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
    SiPixelQualityProbabilities           = cms.untracked.PSet( limit = cms.untracked.int32(-1))
    )

##
## Empty source
##
process.source = cms.Source("EmptySource",
                            firstRun = cms.untracked.uint32(1),
                            numberEventsInRun    = cms.untracked.uint32(1),
                            firstLuminosityBlock = cms.untracked.uint32(1),
                            numberEventsInLuminosityBlock = cms.untracked.uint32(1),
                            )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1))

##
## Database output service
##
process.load("CondCore.CondDB.CondDB_cfi")

##
## Output database (in this case local sqlite file)
##
process.CondDB.connect = 'sqlite_file:SiPixelQualityProbabilities_UltraLegacy2018_v0_mc.db'
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDB,
                                          timetype = cms.untracked.string('runnumber'),
                                          toPut = cms.VPSet(cms.PSet(record = cms.string('SiPixelStatusScenarioProbabilityRcd'),
                                                                     tag = cms.string('SiPixelQualityProbabilities_UltraLegacy2018_v0_mc')
                                                                     )
                                                            )
                                          )


process.WriteInDB = cms.EDAnalyzer("SiPixelQualityProbabilitiesWriteFromASCII",
                                   printDebug    = cms.untracked.bool(False),
                                   record        = cms.string('SiPixelStatusScenarioProbabilityRcd'),
                                   probabilities = cms.string('prob_2018_-1.txt'),
                                   )

process.p = cms.Path(process.WriteInDB)
