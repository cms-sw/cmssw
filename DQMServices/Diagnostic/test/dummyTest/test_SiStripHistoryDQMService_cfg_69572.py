import FWCore.ParameterSet.Config as cms

process = cms.Process("PWRITE")

#########################
# message logger
######################### 

process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring('readFromFile_69572'),
                                    readFromFile_69572 = cms.untracked.PSet(threshold = cms.untracked.string('DEBUG')),
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
                                          outOfOrder = cms.untracked.bool(True),
                                          DBParameters = cms.PSet(
    messageLevel = cms.untracked.int32(2),
    authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
                                          timetype = cms.untracked.string('runnumber'),
                                          connect = cms.string('sqlite_file:dbfile.db'),
                                          toPut = cms.VPSet(cms.PSet(
    record = cms.string("HDQMSummary"),
    tag = cms.string("HDQM_test")
    )),
                                          logconnect = cms.untracked.string("sqlite_file:log.db") 
                                          )

########################
# HistoricDQMService   #
# POPCON Application   #
########################
process.siStripDQMHistoryPopCon = cms.EDAnalyzer("SiStripDQMHistoryPopCon",
        # popcon::PopConAnalyzer
        record = cms.string("HDQMSummary"),
        loggingOn = cms.untracked.bool(True),
        SinceAppendMode = cms.bool(True),

        # SiStripDQMHistoryPopCon
        Source = cms.PSet(
                ## PopCon source handler
                since = cms.untracked.uint32(69572),
                RunNb = cms.uint32(69572),
                iovSequence = cms.untracked.bool(False),
                debug = cms.untracked.bool(False),
                ## DQMStoreReader
                accessDQMFile = cms.bool(True),
                FILE_NAME = cms.untracked.string("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/data/Cosmics__Commissioning08_CRAFT_ALL_V11_227_Tosca090216_ReReco_FromTrackerPointing_v2__RAW-RECO/69572/DQM_V0001_R000069572__Cosmics__Commissioning08_CRAFT_ALL_V11_227_Tosca090216_ReReco_FromTrackerPointing_v2__RAW-RECO_1.root"),
                ## DQMHistoryHelper
                #
                ## base DQM history service
                ME_DIR = cms.untracked.string("Run 69572/SiStrip"),
                histoList = cms.VPSet(
                        # quantities are 'stat', 'landau', 'gauss'
                        # where
                        #'stat' includes entries, mean, rms
                        #'landau' includes
                        #'gauss' includes gaussMean, gaussSigma

                        # CKFTk
                        cms.PSet( keyName = cms.untracked.string("Chi2_CKFTk"), quantitiesToExtract = cms.untracked.vstring("stat"))
                        ,
                        cms.PSet( keyName = cms.untracked.string("NumberOfTracks_CKFTk"), quantitiesToExtract = cms.untracked.vstring("stat"))
                        ,
                        cms.PSet( keyName = cms.untracked.string("NumberOfRecHitsPerTrack_CKFTk"), quantitiesToExtract = cms.untracked.vstring("stat"))

                        # Summary Cluster Properties
                        ,
                        cms.PSet( keyName = cms.untracked.string("Summary_TotalNumberOfClusters_OnTrack"),  quantitiesToExtract = cms.untracked.vstring("stat"))

                        # Summary Cluster properties @ layer level
                        ,
                        cms.PSet( keyName = cms.untracked.string("Summary_TotalNumberOfClusters__OnTrack"),  quantitiesToExtract = cms.untracked.vstring("stat"))
                    )
                ## specific for SiStripDQMHistory
                #
            )
    )

# Schedule
process.p = cms.Path(process.siStripDQMHistoryPopCon)
