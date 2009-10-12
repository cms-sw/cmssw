import FWCore.ParameterSet.Config as cms

process = cms.Process("PWRITE")

##################
# message logger #
##################

process.MessageLogger = cms.Service(
    "MessageLogger",
    destinations = cms.untracked.vstring('readFromFile_RUNNUMBER'),
    readFromFile_RUNNUMBER = cms.untracked.PSet(threshold = cms.untracked.string('DEBUG')),
    debugModules = cms.untracked.vstring('*')
)

#################
# maxEvents ... #
#################

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source(
    "EmptySource",
    timetype = cms.string("runnumber"),
    firstRun = cms.untracked.uint32(1),
    lastRun  = cms.untracked.uint32(1),
    interval = cms.uint32(1)
)

################
# DQM services #
################

process.load("DQMServices.Core.DQM_cfg")

#################
# DB parameters #
#################

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
    toPut = cms.VPSet(cms.PSet(
        record = cms.string("HDQMSummary"),
        tag = cms.string("TAGNAME")
    )),
    logconnect = cms.untracked.string("sqlite_file:log.db") 
)

######################
# POPCON Application #
######################
process.siStripDQMHistoryPopCon = cms.EDAnalyzer(
    "SiStripDQMHistoryPopCon",
    record = cms.string("HDQMSummary"),
    loggingOn = cms.untracked.bool(True),
    SinceAppendMode = cms.bool(True),
    Source = cms.PSet(
        since = cms.untracked.uint32(RUNNUMBER),
        debug = cms.untracked.bool(False)
    )
)

#####################
# HistoryDQMService #
#####################

process.SiStripHistoryDQMService = cms.Service(
    "SiStripHistoryDQMService",
    RunNb = cms.uint32(RUNNUMBER),
    accessDQMFile = cms.bool(True),
    FILE_NAME = cms.untracked.string("FILENAME"),
    ME_DIR = cms.untracked.string("Run RUNNUMBER/Tracking"),
    histoList = cms.VPSet(
    
    # quantities are 'stat', 'landau', 'gauss'
    # where 
    #'stat' includes entries, mean, rms
    #'landau' includes
    #'gauss' includes gaussMean, gaussSigma
    

    #CosmicTk
    cms.PSet( keyName = cms.untracked.string("NumberOfTracks_CosmicTk"), quantitiesToExtract  = cms.untracked.vstring("stat"))
    ,
    cms.PSet( keyName = cms.untracked.string("Chi2overDoF_CosmicTk"), quantitiesToExtract = cms.untracked.vstring("stat"))
    ,
    cms.PSet( keyName = cms.untracked.string("NumberOfRecHitsPerTrack_CosmicTk"), quantitiesToExtract = cms.untracked.vstring("stat"))
    ,
    cms.PSet( keyName = cms.untracked.string("TrackPt_CosmicTk"), quantitiesToExtract = cms.untracked.vstring("stat"))
    ,
    cms.PSet( keyName = cms.untracked.string("TrackPz_CosmicTk"), quantitiesToExtract = cms.untracked.vstring("stat"))
    ,
    cms.PSet( keyName = cms.untracked.string("TrackPx_CosmicTk"), quantitiesToExtract = cms.untracked.vstring("stat"))
    ,
    cms.PSet( keyName = cms.untracked.string("TrackPy_CosmicTk"), quantitiesToExtract = cms.untracked.vstring("stat"))
    ,
    cms.PSet( keyName = cms.untracked.string("TrackPhi_CosmicTk"), quantitiesToExtract = cms.untracked.vstring("stat"))
    ,
    cms.PSet( keyName = cms.untracked.string("TrackEta_CosmicTk"), quantitiesToExtract = cms.untracked.vstring("stat"))
    ,
    cms.PSet( keyName = cms.untracked.string("DistanceOfClosestApproach_CosmicTk"), quantitiesToExtract = cms.untracked.vstring("stat"))

    # CKFTk
    ,
    cms.PSet( keyName = cms.untracked.string("Chi2overDoF_CKFTk"), quantitiesToExtract = cms.untracked.vstring("stat"))
    ,
    cms.PSet( keyName = cms.untracked.string("NumberOfTracks_CKFTk"), quantitiesToExtract = cms.untracked.vstring("stat"))
    ,
    cms.PSet( keyName = cms.untracked.string("NumberOfRecHitsPerTrack_CKFTk"), quantitiesToExtract = cms.untracked.vstring("stat"))
    ,
    cms.PSet( keyName = cms.untracked.string("TrackPt_CKFTk"), quantitiesToExtract = cms.untracked.vstring("stat"))
    ,
    cms.PSet( keyName = cms.untracked.string("TrackPz_CKFTk"), quantitiesToExtract = cms.untracked.vstring("stat"))
    ,
    cms.PSet( keyName = cms.untracked.string("TrackPx_CKFTk"), quantitiesToExtract = cms.untracked.vstring("stat"))
    ,
    cms.PSet( keyName = cms.untracked.string("TrackPy_CKFTk"), quantitiesToExtract = cms.untracked.vstring("stat"))
    ,
    cms.PSet( keyName = cms.untracked.string("TrackPhi_CKFTk"), quantitiesToExtract = cms.untracked.vstring("stat"))
    ,
    cms.PSet( keyName = cms.untracked.string("TrackEta_CKFTk"), quantitiesToExtract = cms.untracked.vstring("stat"))
    ,
    cms.PSet( keyName = cms.untracked.string("DistanceOfClosestApproach_CKFTk"), quantitiesToExtract = cms.untracked.vstring("stat"))

    # RSTk
    ,
    cms.PSet( keyName = cms.untracked.string("Chi2overDoF_RSTk"), quantitiesToExtract = cms.untracked.vstring("stat"))
    ,
    cms.PSet( keyName = cms.untracked.string("NumberOfTracks_RSTk"), quantitiesToExtract = cms.untracked.vstring("stat"))
    ,
    cms.PSet( keyName = cms.untracked.string("NumberOfRecHitsPerTrack_RSTk"), quantitiesToExtract = cms.untracked.vstring("stat"))
    ,
    cms.PSet( keyName = cms.untracked.string("TrackPt_RSTk"), quantitiesToExtract = cms.untracked.vstring("stat"))
    ,
    cms.PSet( keyName = cms.untracked.string("TrackPz_RSTk"), quantitiesToExtract = cms.untracked.vstring("stat"))
    ,
    cms.PSet( keyName = cms.untracked.string("TrackPx_RSTk"), quantitiesToExtract = cms.untracked.vstring("stat"))
    ,
    cms.PSet( keyName = cms.untracked.string("TrackPy_RSTk"), quantitiesToExtract = cms.untracked.vstring("stat"))
    ,
    cms.PSet( keyName = cms.untracked.string("TrackPhi_RSTk"), quantitiesToExtract = cms.untracked.vstring("stat"))
    ,
    cms.PSet( keyName = cms.untracked.string("TrackEta_RSTk"), quantitiesToExtract = cms.untracked.vstring("stat"))
    ,
    cms.PSet( keyName = cms.untracked.string("DistanceOfClosestApproach_RSTk"), quantitiesToExtract = cms.untracked.vstring("stat"))
    
    )
)


# Schedule

process.p = cms.Path(process.siStripDQMHistoryPopCon)




