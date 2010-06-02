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

process.PoolDBOutputService = cms.Service(
    "PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    outOfOrder = cms.untracked.bool(True),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(2),
        authenticationPath = cms.untracked.string('AUTHENTICATIONPATH')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('DATABASE'),
    toPut = cms.VPSet(
        cms.PSet(
            record = cms.string("HDQMSummary"),
            tag = cms.string("TAGNAME")
        )
    ),
    logconnect = cms.untracked.string("sqlite_file:log.db") 
)

########################
# POPCON Application
########################
process.siPixelDQMHistoryPopCon = cms.EDAnalyzer(
    "SiPixelDQMHistoryPopCon",
    record = cms.string("HDQMSummary"),
    loggingOn = cms.untracked.bool(True),
    SinceAppendMode = cms.bool(True),
    Source = cms.PSet(since = cms.untracked.uint32(RUNNUMBER),debug = cms.untracked.bool(False))
)

########################
# HistoricDQMService
########################

process.SiPixelHistoryDQMService = cms.Service(
    "SiPixelHistoryDQMService",
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
      # cms.PSet( keyName = cms.untracked.string("SUMOFF_adc"), quantitiesToExtract = cms.untracked.vstring("user") ),
      # cms.PSet( keyName = cms.untracked.string("SUMOFF_charge_OffTrack"), quantitiesToExtract = cms.untracked.vstring("user") ),
      # cms.PSet( keyName = cms.untracked.string("SUMOFF_charge_OnTrack" ), quantitiesToExtract = cms.untracked.vstring("user") ),
      # cms.PSet( keyName = cms.untracked.string("SUMOFF_nRecHits"), quantitiesToExtract = cms.untracked.vstring("user") ),
      # cms.PSet( keyName = cms.untracked.string("SUMOFF_nclusters_OffTrack"), quantitiesToExtract = cms.untracked.vstring("user") ),
      # cms.PSet( keyName = cms.untracked.string("SUMOFF_nclusters_OnTrack" ), quantitiesToExtract = cms.untracked.vstring("user") ),
      # cms.PSet( keyName = cms.untracked.string("SUMOFF_ndigis"), quantitiesToExtract = cms.untracked.vstring("user") ),
      # cms.PSet( keyName = cms.untracked.string("SUMOFF_size_OffTrack"), quantitiesToExtract = cms.untracked.vstring("user") ),
      # cms.PSet( keyName = cms.untracked.string("SUMOFF_size_OnTrack" ), quantitiesToExtract = cms.untracked.vstring("user") ),
      cms.PSet( keyName = cms.untracked.string("SUMOFF_adc"), quantitiesToExtract = cms.untracked.vstring("user_ymean") ),
      cms.PSet( keyName = cms.untracked.string("SUMOFF_charge_OffTrack"), quantitiesToExtract = cms.untracked.vstring("user_ymean") ),
      cms.PSet( keyName = cms.untracked.string("SUMOFF_charge_OnTrack" ), quantitiesToExtract = cms.untracked.vstring("user_ymean") ),
      cms.PSet( keyName = cms.untracked.string("SUMOFF_nRecHits"), quantitiesToExtract = cms.untracked.vstring("user_ymean") ),
      cms.PSet( keyName = cms.untracked.string("SUMOFF_nclusters_OffTrack"), quantitiesToExtract = cms.untracked.vstring("user_ymean") ),
      cms.PSet( keyName = cms.untracked.string("SUMOFF_nclusters_OnTrack" ), quantitiesToExtract = cms.untracked.vstring("user_ymean") ),
      cms.PSet( keyName = cms.untracked.string("SUMOFF_ndigis"), quantitiesToExtract = cms.untracked.vstring("user_ymean") ),
      cms.PSet( keyName = cms.untracked.string("SUMOFF_size_OffTrack"), quantitiesToExtract = cms.untracked.vstring("user_ymean") ),
      cms.PSet( keyName = cms.untracked.string("SUMOFF_size_OnTrack" ), quantitiesToExtract = cms.untracked.vstring("user_ymean") ),
      cms.PSet( keyName = cms.untracked.string("ntracks_rsWithMaterialTracksP5" ), quantitiesToExtract = cms.untracked.vstring("user_A") ),
      cms.PSet( keyName = cms.untracked.string("ntracks_rsWithMaterialTracksP5" ), quantitiesToExtract = cms.untracked.vstring("user_B") )
      #cms.PSet( keyName = cms.untracked.string("ntracks_rsWithMaterialTracksP5" ), quantitiesToExtract = cms.untracked.vstring("userB") ), # pixel/All, FPix/BPix
      )
    )


# Schedule

process.p = cms.Path(process.siPixelDQMHistoryPopCon)




