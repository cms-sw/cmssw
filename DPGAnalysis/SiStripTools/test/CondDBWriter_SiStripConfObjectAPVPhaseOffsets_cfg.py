# The following comments couldn't be translated into the new config version:

# upload to database 

#string timetype = "timestamp"    

import FWCore.ParameterSet.Config as cms

process = cms.Process("CondDBWriterAPVPhaseOffsets")

process.MessageLogger = cms.Service(
    "MessageLogger",
    debugModules = cms.untracked.vstring('siStripConfObjectDummyDBWriter'),
    threshold = cms.untracked.string('INFO'),
    destinations = cms.untracked.vstring('ConfObjectBuilder.log')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1),
)

process.load("DPGAnalysis.SiStripTools.SiStripConfObjectAPVPhaseOffsetsFakeESSource_cfi")
process.load("CalibTracker.SiStripESProducers.DBWriter.SiStripConfObjectDummyDBWriter_cfi")

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(2),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:apvphaseoffsets_forHLT.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiStripConfObjectRcd'),
        tag = cms.string('SiStripAPVPhaseOffsets_real_v1')
    ))
)

process.siStripConfObjectDummyDBWriter.record=process.PoolDBOutputService.toPut[0].record 
process.siStripConfObjectDummyDBWriter.label=cms.untracked.string("apvphaseoffsets")

# This specifies the IOV, not the EmptySource (to see why look at the DummyDBWriter code)
process.siStripConfObjectDummyDBWriter.OpenIovAt = ""
process.siStripConfObjectDummyDBWriter.OpenIovAtTime = 237426

process.p1 = cms.Path(process.siStripConfObjectDummyDBWriter)
