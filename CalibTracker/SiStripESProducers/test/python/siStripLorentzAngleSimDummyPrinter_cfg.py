# The following comments couldn't be translated into the new config version:

# upload to database 

#string timetype = "timestamp"    

import FWCore.ParameterSet.Config as cms

process = cms.Process("Reader")

process.MessageLogger = cms.Service("MessageLogger",
    LorentzAngleReaderDebug = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG')
    ),
    LorentzAngleReaderSummary = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    debugModules = cms.untracked.vstring('*'),
    files = cms.untracked.PSet(
        LorentzAngleSimReaderDebug = cms.untracked.PSet(

        ),
        LorentzAngleSimReaderSummary = cms.untracked.PSet(

        )
    )
)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(100000)
)

process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'IDEAL_30X::All'

#process.poolDBESSource = cms.ESSource("PoolDBESSource",
#   BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
#   DBParameters = cms.PSet(
#        messageLevel = cms.untracked.int32(2),
#        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
#    ),
#    timetype = cms.untracked.string('runnumber'),
#    connect = cms.string('sqlite_file:dbfile.db'),
#    toGet = cms.VPSet(cms.PSet(
#        record = cms.string('SiStripLorentzAngleSimRcd'),
#        tag = cms.string('SiStripLorentzAngle_IdealSim_31X')
#    ))
#)

process.reader = cms.EDFilter("SiStripLorentzAngleSimDummyPrinter")
                              
process.p1 = cms.Path(process.reader)


