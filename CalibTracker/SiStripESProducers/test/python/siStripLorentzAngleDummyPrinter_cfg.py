# The following comments couldn't be translated into the new config version:

# upload to database 

#string timetype = "timestamp"    

import FWCore.ParameterSet.Config as cms

process = cms.Process("Reader")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring("*"),
    LorentzAngleReaderSummary = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    LorentzAngleReaderDebug = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG')
    ),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    destinations = cms.untracked.vstring('LorentzAngleReaderSummary', 'LorentzAngleReaderDebug')
)

# Use this instead to see only the summary
#process.MessageLogger = cms.Service("MessageLogger",
#    debugModules = cms.untracked.vstring(''),
#    LorentzAngleReaderSummary = cms.untracked.PSet(
#        threshold = cms.untracked.string('INFO')
#    ),
#    cout = cms.untracked.PSet(
#        threshold = cms.untracked.string('INFO')
#    ),
#    destinations = cms.untracked.vstring('LorentzAngleReaderSummary')
#)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(100000)
)

process.poolDBESSource = cms.ESSource("PoolDBESSource",
   BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
   DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(2),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:dbfile.db'),
#    connect = cms.string('oracle://cms_orcoff_prod/CMS_COND_21X_STRIP'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('SiStripLorentzAngleRcd'),
#        tag = cms.string('SiStripNoise_CRAFT_21X_v4_offline')
        tag = cms.string('SiStripLorentzAngle_Fake_30X')
    ))
)

process.reader = cms.EDFilter("SiStripLorentzAngleDummyPrinter")
                              
process.p1 = cms.Path(process.reader)


