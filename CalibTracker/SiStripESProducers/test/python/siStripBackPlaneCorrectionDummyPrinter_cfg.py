# The following comments couldn't be translated into the new config version:

# upload to database 

#string timetype = "timestamp"    

import FWCore.ParameterSet.Config as cms

process = cms.Process("Reader")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('BackPlaneCorrectionReaderSummary'),
    threshold = cms.untracked.string('DEBUG'),
    destinations = cms.untracked.vstring('BackPlaneCorrectionReader.log')
)


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
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('SiStripBackPlaneCorrectionRcd'),
        tag = cms.string('SiStripBackPlaneCorrection_Ideal_31X')
    ))
)

process.reader = cms.EDAnalyzer("SiStripBackPlaneCorrectionDummyPrinter")
                              
process.p1 = cms.Path(process.reader)


