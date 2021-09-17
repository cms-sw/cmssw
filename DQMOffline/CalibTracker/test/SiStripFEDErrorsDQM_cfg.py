import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

#########################
# message logger
######################### 

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        noLineBreaks = cms.untracked.bool(False),
        threshold = cms.untracked.string('ERROR')
    ),
    debugModules = cms.untracked.vstring('*'),
    files = cms.untracked.PSet(
        debug = cms.untracked.PSet(
            noLineBreaks = cms.untracked.bool(False),
            threshold = cms.untracked.string('DEBUG')
        ),
        error = cms.untracked.PSet(
            noLineBreaks = cms.untracked.bool(False),
            threshold = cms.untracked.string('ERROR')
        ),
        info = cms.untracked.PSet(
            noLineBreaks = cms.untracked.bool(False),
            threshold = cms.untracked.string('INFO')
        ),
        warning = cms.untracked.PSet(
            noLineBreaks = cms.untracked.bool(False),
            threshold = cms.untracked.string('WARNING')
        )
    )
)

#process.MessageLogger = cms.Service("MessageLogger"
#destinations = cms.untracked.vstring('cout', 'readFromFile'),
#readFromFile = cms.untracked.PSet(
#    threshold = cms.untracked.string('DEBUG')),
#debugModules = cms.untracked.vstring('*')
#)


#########################
# maxEvents ...
#########################

process.maxEvents = cms.untracked.PSet(
     input = cms.untracked.int32(1))

# process.source = cms.Source("EmptySource",
#     timetype = cms.string("runnumber"),
#     firstRun = cms.untracked.uint32(118066),
#     lastRun  = cms.untracked.uint32(118066),
#     interval = cms.uint32(1)
# )
process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(123815)
)

#########################
# DQM services
#########################
process.load("DQMServices.Core.DQM_cfg")
#from DQMServices.Core.DQM_cfg import *

########################
#Global tag
########################
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = "CRAFT0831X_V1::All"
#process.GlobalTag.globaltag = "GR09_31X_V1P::All"
process.GlobalTag.globaltag = "GR09_E_V2::All"
process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')

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


# ########################
# # POPCON Application
# ########################
# process.siStripPopConFEDErrorsDQM = cms.OutputModule(
#     "SiStripPopConFEDErrorsDQM",
#     record = cms.string("SiStripBadStripRcd"),
#     loggingOn = cms.untracked.bool(True),
#     SinceAppendMode = cms.bool(True),
#     Source = cms.PSet(
#         since = cms.untracked.uint32(108298),
#         debug = cms.untracked.bool(False)
#         )
#     )


############################################
# SiStripFEDErrorsDQM (POPCON Application) #
############################################

process.siStripFEDErrorsDQM = cms.EDAnalyzer("SiStripPopConFEDErrorsDQM",
                                             RunNb = cms.uint32(123815),
                                             ##accessDQMFile = cms.bool(True),
                                             ##FILE_NAME = cms.untracked.string("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/data/Online/123/815/DQM_V0001_SiStrip_R000123815.root"),
                                             ##ME_DIR = cms.untracked.string("Run 123815"),
                                             ##histoList = cms.VPSet(),
                                             Threshold = cms.untracked.double(0.0),
                                             Debug = cms.untracked.uint32(1)
                                             )

# Schedule

process.p = cms.Path(process.siStripFEDErrorsDQM)
process.asciiPrint = cms.OutputModule("AsciiOutputModule") 
process.ep = cms.EndPath(process.asciiPrint)
    



