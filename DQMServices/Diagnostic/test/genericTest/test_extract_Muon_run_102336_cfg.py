import FWCore.ParameterSet.Config as cms

process = cms.Process("PWRITE")

#########################
# message logger
######################### 

process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring('readFromFile_102336'),
                                    readFromFile_102336 = cms.untracked.PSet(threshold = cms.untracked.string('DEBUG')),
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

#########################################
# HistoricDQMService POPCON Application #
#########################################
process.genericDQMHistoryPopCon = cms.EDAnalyzer("GenericDQMHistoryPopCon",
        # popcon::PopConAnalyzer
        record = cms.string("HDQMSummary"),
        loggingOn = cms.untracked.bool(True),
        SinceAppendMode = cms.bool(True),
        # GenericDQMHistoryPopCon
        Source = cms.PSet(
                ## PopCon source handler
                since = cms.untracked.uint32(102336),
                RunNb = cms.uint32(102336),
                iovSequence = cms.untracked.bool(False),
                debug = cms.untracked.bool(False),
                ## DQMStoreReader
                accessDQMFile = cms.bool(True),
                FILE_NAME = cms.untracked.string("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/data/PromptReco/102/336/DQM_V0001_R000102336__Cosmics__Commissioning09-PromptReco-v4__RECO.root"),
                ## DQMHistoryHelper
                #
                ## base DQM history service
                ME_DIR = cms.untracked.string("Run 102336/Muons/Run summary/MuonRecoAnalyzer"),
                histoList = cms.VPSet(
                        # quantities are 'stat', 'landau', 'gauss'
                        # where
                        #'stat' includes entries, mean, rms
                        #'landau' includes
                        #'gauss' includes gaussMean, gaussSigma

                        # CKFTk
                        cms.PSet( keyName = cms.untracked.string("StaMuon_p"), quantitiesToExtract = cms.untracked.vstring("stat","user")),
                        cms.PSet( keyName = cms.untracked.string("StaMuon_pt"), quantitiesToExtract = cms.untracked.vstring("stat","user")),
                        cms.PSet( keyName = cms.untracked.string("StaMuon_q"), quantitiesToExtract = cms.untracked.vstring("stat","user")),
                        cms.PSet( keyName = cms.untracked.string("StaMuon_eta"), quantitiesToExtract = cms.untracked.vstring("stat")),
                        cms.PSet( keyName = cms.untracked.string("StaMuon_theta"), quantitiesToExtract = cms.untracked.vstring("stat")),
                        cms.PSet( keyName = cms.untracked.string("StaMuon_phi"), quantitiesToExtract = cms.untracked.vstring("stat"))
                        #cms.PSet( keyName = cms.untracked.string("GlbMuon_Glb_p"), quantitiesToExtract = cms.untracked.vstring("stat","user")),
                        #cms.PSet( keyName = cms.untracked.string("GlbMuon_Glb_pt"), quantitiesToExtract = cms.untracked.vstring("stat","user")),
                        #cms.PSet( keyName = cms.untracked.string("GlbMuon_Glb_q"), quantitiesToExtract = cms.untracked.vstring("stat","user")),
                        #cms.PSet( keyName = cms.untracked.string("GlbMuon_Glb_eta"), quantitiesToExtract = cms.untracked.vstring("stat")),
                        #cms.PSet( keyName = cms.untracked.string("GlbMuon_Glb_theta"), quantitiesToExtract = cms.untracked.vstring("stat")),
                        #cms.PSet( keyName = cms.untracked.string("GlbMuon_Glb_phi"), quantitiesToExtract = cms.untracked.vstring("stat"))
                    ),
                ## specific for GenericDQMHistory
                DetectorId = cms.uint32(1),
            )
    )

# Schedule

process.p = cms.Path(process.genericDQMHistoryPopCon)




