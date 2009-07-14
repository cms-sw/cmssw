import FWCore.ParameterSet.Config as cms

process = cms.Process("PWRITE")

#########################
# message logger
######################### 

process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring('cout'),
#readFromFile_RUNNUMBER = cms.untracked.PSet(threshold = cms.untracked.string('DEBUG')),
                                    debugModules = cms.untracked.vstring('*')
                                    )


#########################
# maxEvents ...
#########################

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1))

process.source = cms.Source("EmptySource",
                            timetype = cms.string("runnumber"),
                            firstRun = cms.untracked.uint32(1),
                            lastRun  = cms.untracked.uint32(1),
                            interval = cms.uint32(1)
                            )

#########################
# DQM services
#########################

process.load("DQMServices.Core.DQM_cfg")


########################
# DB parameters
########################

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
                                          DBParameters = cms.PSet(
    messageLevel = cms.untracked.int32(2),
    authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
                                          timetype = cms.untracked.string('runnumber'),
                                          connect = cms.string('sqlite_file:dbfile.db'),
                                          toPut = cms.VPSet(cms.PSet(
    record = cms.string("HDQMSummary"),
    tag = cms.string("TAGNAME")
    )),
                                          logconnect = cms.untracked.string("sqlite_file:log.db") 
                                          )

########################
# POPCON Application
########################
process.siPixelDQMHistoryPopCon = cms.EDAnalyzer("SiPixelDQMHistoryPopCon",
                                                 record = cms.string("HDQMSummary"),
                                                 loggingOn = cms.untracked.bool(True),
                                                 SinceAppendMode = cms.bool(True),
                                                 Source = cms.PSet(since = cms.untracked.uint32(RUNNUMBER),debug = cms.untracked.bool(False))
                                                 ) 


########################
# HistoricDQMService
########################

process.SiPixelHistoryDQMService = cms.Service("SiPixelHistoryDQMService",
    RunNb = cms.uint32(RUNNUMBER),
    accessDQMFile = cms.bool(True),
    FILE_NAME = cms.untracked.string("FILENAME"),
    ME_DIR = cms.untracked.string("Run RUNNUMBER"),
    histoList = cms.VPSet(

# quantities are 'stat', 'landau', 'gauss'
# where 
#'stat' includes entries, mean, rms
#'landau' includes
#'gauss' includes gaussMean, gaussSigma


# CKFTk
      cms.PSet( keyName = cms.untracked.string("SUMOFF_ClustX"), quantitiesToExtract = cms.untracked.vstring("user") ),
      cms.PSet( keyName = cms.untracked.string("SUMOFF_ClustY"), quantitiesToExtract = cms.untracked.vstring("user") ),
      cms.PSet( keyName = cms.untracked.string("SUMOFF_adc"), quantitiesToExtract = cms.untracked.vstring("user") ),
      cms.PSet( keyName = cms.untracked.string("SUMOFF_charge"), quantitiesToExtract = cms.untracked.vstring("user") ),
      cms.PSet( keyName = cms.untracked.string("SUMOFF_nRecHits"), quantitiesToExtract = cms.untracked.vstring("user") ),
      cms.PSet( keyName = cms.untracked.string("SUMOFF_nclusters"), quantitiesToExtract = cms.untracked.vstring("user") ),
      cms.PSet( keyName = cms.untracked.string("SUMOFF_ndigis"), quantitiesToExtract = cms.untracked.vstring("user") ),
      cms.PSet( keyName = cms.untracked.string("SUMOFF_sizeX"), quantitiesToExtract = cms.untracked.vstring("user") ),
      cms.PSet( keyName = cms.untracked.string("SUMOFF_sizeY"), quantitiesToExtract = cms.untracked.vstring("user") ),
      cms.PSet( keyName = cms.untracked.string("SUMOFF_size"), quantitiesToExtract = cms.untracked.vstring("user") ),
      cms.PSet( keyName = cms.untracked.string("SUMOFF_x"), quantitiesToExtract = cms.untracked.vstring("user") ),
      cms.PSet( keyName = cms.untracked.string("SUMOFF_y"), quantitiesToExtract = cms.untracked.vstring("user") ),
      cms.PSet( keyName = cms.untracked.string("SUMOFF_maxrow"), quantitiesToExtract = cms.untracked.vstring("user") ),
      cms.PSet( keyName = cms.untracked.string("SUMOFF_minrow"), quantitiesToExtract = cms.untracked.vstring("user") ),
      cms.PSet( keyName = cms.untracked.string("SUMOFF_maxcol"), quantitiesToExtract = cms.untracked.vstring("user") ),
      cms.PSet( keyName = cms.untracked.string("SUMOFF_mincol"), quantitiesToExtract = cms.untracked.vstring("user") )
      )
    )


# Schedule

process.p = cms.Path(process.siPixelDQMHistoryPopCon)




