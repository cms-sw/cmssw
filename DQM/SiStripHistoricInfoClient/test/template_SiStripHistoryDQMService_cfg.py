import FWCore.ParameterSet.Config as cms

process = cms.Process("PWRITE")

##################
# message logger #
##################

process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring('readFromFile_RUNNUMBER'),
                                    readFromFile_RUNNUMBER = cms.untracked.PSet(threshold = cms.untracked.string('DEBUG')),
                                    debugModules = cms.untracked.vstring('*')
                                    )


#################
# maxEvents ... #
#################

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1))

process.source = cms.Source("EmptySource",
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
        # authenticationPath = cms.untracked.string('/home/cmstacuser/historyDQM/Cron/Scripts/Authentication')
        authenticationPath = cms.untracked.string('AUTHENTICATIONPATH')
    ),
    timetype = cms.untracked.string('runnumber'),
    # connect = cms.string('sqlite_file:dbfile.db'),
    # connect = cms.string('oracle://cms_orcoff_prep/CMS_COND_STRIP'),
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
    ME_DIR = cms.untracked.string("Run RUNNUMBER/SiStrip"),
    histoList = cms.VPSet(
    
    # quantities are 'stat', 'landau', 'gauss'
    # where 
    #'stat' includes entries, mean, rms
    #'landau' includes
    #'gauss' includes gaussMean, gaussSigma
    

    # Summary Cluster Properties (subdet tag _in_TOB)
    cms.PSet( keyName = cms.untracked.string("Summary_TotalNumberOfClusters_OffTrack"), quantitiesToExtract = cms.untracked.vstring("stat"))
    ,
    cms.PSet( keyName = cms.untracked.string("Summary_TotalNumberOfClusters_OnTrack"), quantitiesToExtract = cms.untracked.vstring("stat"))
    ,
    cms.PSet( keyName = cms.untracked.string("Summary_ClusterChargeCorr_OnTrack"), quantitiesToExtract = cms.untracked.vstring("stat","landau"))
    ,
    cms.PSet( keyName = cms.untracked.string("Summary_ClusterCharge_OffTrack"), quantitiesToExtract = cms.untracked.vstring("stat","landau"))
    ,
    cms.PSet( keyName = cms.untracked.string("Summary_ClusterNoise_OnTrack"), quantitiesToExtract = cms.untracked.vstring("stat","gauss"))
    ,
    cms.PSet( keyName = cms.untracked.string("Summary_ClusterNoise_OffTrack"), quantitiesToExtract = cms.untracked.vstring("stat","gauss"))
    ,
    cms.PSet( keyName = cms.untracked.string("Summary_ClusterStoNCorr_OnTrack"), quantitiesToExtract = cms.untracked.vstring("stat","landau"))
    ,
    cms.PSet( keyName = cms.untracked.string("Summary_ClusterStoN_OffTrack"), quantitiesToExtract = cms.untracked.vstring("stat","landau"))
    ,
    cms.PSet( keyName = cms.untracked.string("Summary_ClusterWidth_OnTrack"), quantitiesToExtract = cms.untracked.vstring("stat"))
    ,
    cms.PSet( keyName = cms.untracked.string("Summary_ClusterWidth_OffTrack"), quantitiesToExtract = cms.untracked.vstring("stat"))
    ,
    # Summary Cluster properties @ layer level (layer tag __TOB__layer__1)
    cms.PSet( keyName = cms.untracked.string("Summary_TotalNumberOfDigis"), quantitiesToExtract = cms.untracked.vstring("stat","landau"))
    ,
    cms.PSet( keyName = cms.untracked.string("Summary_ClusterChargeCorr__OnTrack"), quantitiesToExtract = cms.untracked.vstring("stat","landau"))
    ,
    cms.PSet( keyName = cms.untracked.string("Summary_ClusterCharge__OffTrack"), quantitiesToExtract = cms.untracked.vstring("stat","landau"))
    ,
    cms.PSet( keyName = cms.untracked.string("Summary_ClusterNoise__OnTrack"), quantitiesToExtract = cms.untracked.vstring("stat","gauss"))
    ,
    cms.PSet( keyName = cms.untracked.string("Summary_ClusterNoise__OffTrack"), quantitiesToExtract = cms.untracked.vstring("stat","gauss"))
    ,
    cms.PSet( keyName = cms.untracked.string("Summary_ClusterStoNCorr__OnTrack"), quantitiesToExtract = cms.untracked.vstring("stat","landau"))
    ,
    cms.PSet( keyName = cms.untracked.string("Summary_ClusterStoN__OffTrack"), quantitiesToExtract = cms.untracked.vstring("stat","landau"))
    ,
    cms.PSet( keyName = cms.untracked.string("Summary_ClusterWidth__OnTrack"), quantitiesToExtract = cms.untracked.vstring("stat"))
    ,
    cms.PSet( keyName = cms.untracked.string("Summary_ClusterWidth__OffTrack"), quantitiesToExtract = cms.untracked.vstring("stat"))
    ,
    # FED errors:
    cms.PSet( keyName = cms.untracked.string("nFEDErrors"), quantitiesToExtract = cms.untracked.vstring("stat"))
    ,
    cms.PSet( keyName = cms.untracked.string("nBadActiveChannelStatusBits"), quantitiesToExtract = cms.untracked.vstring("stat"))
    ,
    cms.PSet( keyName = cms.untracked.string("nBadChannelStatusBits"), quantitiesToExtract = cms.untracked.vstring("stat"))
    ,
    cms.PSet( keyName = cms.untracked.string("nAPVAddressError"), quantitiesToExtract = cms.untracked.vstring("stat"))
    ,
    cms.PSet( keyName = cms.untracked.string("nUnlocked"), quantitiesToExtract = cms.untracked.vstring("stat"))
    ,
    cms.PSet( keyName = cms.untracked.string("nOutOfSync"), quantitiesToExtract = cms.untracked.vstring("stat"))

    )
)

# Schedule

process.p = cms.Path(process.siStripDQMHistoryPopCon)




