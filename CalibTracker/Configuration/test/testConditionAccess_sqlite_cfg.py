# The following comments couldn't be translated into the new config version:

# upload to database 

#string timetype = "timestamp"    

import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    debugModules = cms.untracked.vstring(''),
    files = cms.untracked.PSet(
        testConditionAccess = cms.untracked.PSet(

        )
    ),
    testConditionAccess = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

process.poolDBESSource = cms.ESSource("PoolDBESSource",
   BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
   DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(2),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:dbfile.db'),
    toGet = cms.VPSet(
    cms.PSet(record = cms.string('SiStripFedCablingRcd'),tag = cms.string('SiStripFedCabling_30X')),
#    cms.PSet(record = cms.string('SiStripNoiseRcd'),tag = cms.string('SiStripNoise_Fake_PeakMode_30X'))
                      )
)


process.fedcablingreader = cms.EDAnalyzer("SiStripFedCablingReader")
#process.noise            = cms.EDFilter("SiStripNoisesReader", printDebug = cms.untracked.uint32(4))

process.p1 = cms.Path(process.fedcablingreader) #*process.noise)


