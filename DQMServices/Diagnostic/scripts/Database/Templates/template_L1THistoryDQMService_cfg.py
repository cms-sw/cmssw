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
         ## authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
         authenticationPath = cms.untracked.string('AUTHENTICATIONPATH')
    ),

    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('DATABASE'),
    ##connect = cms.string('sqlite_file:test_L1T_HDQM_Online_dbfile.db'),
    toPut = cms.VPSet(cms.PSet(
         record = cms.string("HDQMSummary"),
         tag = cms.string("TAGNAME")
    )),
    logconnect = cms.untracked.string("sqlite_file:log.db") 
    ##logconnect = cms.untracked.string("sqlite_file:test_L1T_HDQM_Online_log.db") 
)

######################
# POPCON Application #
######################

process.l1tDQMHistoryPopCon = cms.EDAnalyzer(
    "L1TDQMHistoryPopCon",
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

process.L1THistoryDQMService = cms.Service(
    "L1THistoryDQMService",
    RunNb = cms.uint32(RUNNUMBER),
    accessDQMFile = cms.bool(True),
    FILE_NAME = cms.untracked.string("FILENAME"),
    ME_DIR = cms.untracked.string("Run RUNNUMBER/L1T/Run summary/"),
    threshold = cms.untracked.double(0),
    histoList = cms.VPSet(

    cms.PSet( keyName = cms.untracked.string('EtHad'), quantitiesToExtract = cms.untracked.vstring("stat"))
    ,
    cms.PSet( keyName = cms.untracked.string('EtHad'), quantitiesToExtract = cms.untracked.vstring("usrMean"))
    ,
    cms.PSet( keyName = cms.untracked.string('EtMiss'), quantitiesToExtract = cms.untracked.vstring("stat"))
    ,
    cms.PSet( keyName = cms.untracked.string('EtMiss'), quantitiesToExtract = cms.untracked.vstring("usrMean"))
    ,
    cms.PSet( keyName = cms.untracked.string('EtTotal'), quantitiesToExtract = cms.untracked.vstring("stat"))
    ,
    cms.PSet( keyName = cms.untracked.string('EtTotal'), quantitiesToExtract = cms.untracked.vstring("usrMean"))
    ,
    cms.PSet( keyName = cms.untracked.string('HtMiss'), quantitiesToExtract = cms.untracked.vstring("stat"))
    ,
    cms.PSet( keyName = cms.untracked.string('HtMiss'), quantitiesToExtract = cms.untracked.vstring("usrMean"))
    ,
    cms.PSet( keyName = cms.untracked.string('GMT_pt'), quantitiesToExtract = cms.untracked.vstring("stat"))
    ,
    cms.PSet( keyName = cms.untracked.string('GMT_pt'), quantitiesToExtract = cms.untracked.vstring("usrMean"))

    )
)

process.p = cms.Path(process.l1tDQMHistoryPopCon)

