import FWCore.ParameterSet.Config as cms
process = cms.Process("ProcessOne")

##
## MessageLogger
##
process.load('FWCore.MessageService.MessageLogger_cfi')   
process.MessageLogger.cerr.enable = False
process.MessageLogger.SiPixelQualityProbabilitiesTestWriter=dict()  
process.MessageLogger.SiPixelQualityProbabilities=dict()  
process.MessageLogger.cout = cms.untracked.PSet(
    enable    = cms.untracked.bool(True),
    enableStatistics = cms.untracked.bool(True),
    threshold = cms.untracked.string("INFO"),
    default   = cms.untracked.PSet(limit = cms.untracked.int32(0)),                       
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1),
                                   reportEvery = cms.untracked.int32(1000)
                                   ),                                                      
    SiPixelQualityProbabilitiesTestWriter = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
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
process.CondDB.connect = 'sqlite_file:SiPixelStatusScenarioProbabilities.db'
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDB,
                                          timetype = cms.untracked.string('runnumber'),
                                          toPut = cms.VPSet(cms.PSet(record = cms.string('SiPixelStatusScenarioProbabilityRcd'),
                                                                     tag = cms.string('SiPixelQualityProbabilities_v0_mc')
                                                                     )
                                                            )
                                          )


process.WriteInDB = cms.EDAnalyzer("SiPixelQualityProbabilitiesTestWriter",
                                   printDebug    = cms.untracked.bool(True),
                                   record        = cms.string('SiPixelStatusScenarioProbabilityRcd'),
                                   probabilities = cms.string('snapshotProb_322633.txt'),
                                   snapshots     = cms.string('snapshot_ids.txt')
                                   )

process.p = cms.Path(process.WriteInDB)
