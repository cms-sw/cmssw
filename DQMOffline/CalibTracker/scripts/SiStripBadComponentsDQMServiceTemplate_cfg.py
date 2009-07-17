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
connect = cms.string("sqlite_file:dbfile.db"),
timetype = cms.untracked.string("runnumber"),
DBParameters = cms.PSet(
   authenticationPath = cms.untracked.string("/afs/cern.ch/cms/DB/conddb"),
   messageLevel = cms.untracked.int32(4)
),
toPut = cms.VPSet(
    cms.PSet(
        record = cms.string("SiStripBadStripRcd"),
        tag = cms.string("SiStripBadStrip_test1")
    )
),
logconnect = cms.untracked.string("sqlite_file:log.db") 
)


########################
# POPCON Application
########################
process.siStripPopConBadComponentsDQM = cms.OutputModule("SiStripPopConBadComponentsDQM",
record = cms.string("SiStripBadStripRcd"),
loggingOn = cms.untracked.bool(True),
SinceAppendMode = cms.bool(True),
Source = cms.PSet(
   since = cms.untracked.uint32(RUNNUMBER),
   debug = cms.untracked.bool(False))
) 


##########################
# BadComponentsDQMService
##########################

process.SiStripBadComponentsDQMService = cms.Service("SiStripBadComponentsDQMService",
                                                     RunNb = cms.uint32(RUNNUMBER),
                                                     accessDQMFile = cms.bool(True),
                                                     FILE_NAME = cms.untracked.string("FILENAME"),
                                                     ME_DIR = cms.untracked.string("Run RUNNUMBER"),
                                                     histoList = cms.VPSet()
                                                     )

# Schedule

process.p = cms.Path(process.siStripPopConBadComponentsDQM)
process.asciiPrint = cms.OutputModule("AsciiOutputModule")
process.ep = cms.EndPath(process.asciiPrint)





