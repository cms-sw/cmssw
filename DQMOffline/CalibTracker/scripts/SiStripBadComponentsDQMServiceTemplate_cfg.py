import FWCore.ParameterSet.Config as cms

process = cms.Process("PWRITE")

#########################
# message logger
######################### 

process.MessageLogger = cms.Service("MessageLogger",
destinations = cms.untracked.vstring('cout', 'readFromFile_RUNNUMBER'),
readFromFile_RUNNUMBER = cms.untracked.PSet(
    threshold = cms.untracked.string('DEBUG')),
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
#from DQMServices.Core.DQM_cfg import *


########################
# DB parameters
########################

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
#connect = cms.string("sqlite_file:historicDQM.db"),
BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
connect = cms.string("oracle://cms_orcon_prod/CMS_COND_31X_STRIP"),
timetype = cms.untracked.string("runnumber"),
DBParameters = cms.PSet(
   authenticationPath = cms.untracked.string("/nfshome0/popcondev/conddb"),
   messageLevel = cms.untracked.int32(1)
),
toPut = cms.VPSet(
    cms.PSet(
        record = cms.string("SiStripBadStripRcd"),
        tag = cms.string("SiStripBadStrip_FromOnlineDQM_V2")
    )
),
logconnect = cms.untracked.string("oracle://cms_orcon_prod/CMS_COND_31X_POPCONLOG") 
)


########################
# POPCON Application
########################
process.siStripPopConBadComponentsDQM = cms.EDAnalyzer("SiStripPopConBadComponentsDQM",
    record = cms.string("SiStripBadStripRcd"),
    loggingOn = cms.untracked.bool(True),
    SinceAppendMode = cms.bool(True),
    Source = cms.PSet(
        since = cms.untracked.uint32(RUNNUMBER),
        debug = cms.untracked.bool(False),
        ######################
        ## BadComponentsDQM ##
        ######################
        RunNb = cms.uint32(RUNNUMBER),
        accessDQMFile = cms.bool(True),
        FILE_NAME = cms.untracked.string("FILENAME"),
        ME_DIR = cms.untracked.string("Run RUNNUMBER"),
        histoList = cms.VPSet()
    )
)

# Schedule

process.p = cms.Path(process.siStripPopConBadComponentsDQM)
process.asciiPrint = cms.OutputModule("AsciiOutputModule")
process.ep = cms.EndPath(process.asciiPrint)





