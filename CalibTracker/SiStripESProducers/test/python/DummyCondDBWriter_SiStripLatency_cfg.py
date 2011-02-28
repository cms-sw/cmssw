# The following comments couldn't be translated into the new config version:

# upload to database 

#string timetype = "timestamp"    

import FWCore.ParameterSet.Config as cms

process = cms.Process("Builder")

process.MessageLogger = cms.Service(
    "MessageLogger",
    debugModules = cms.untracked.vstring('siStripLatencyDummyDBWriter'),
    threshold = cms.untracked.string('INFO'),
    destinations = cms.untracked.vstring('LatencyBuilder.log')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1),
)

process.load("CalibTracker.SiStripESProducers.fake.SiStripLatencyFakeESSource_cfi")
process.load("CalibTracker.SiStripESProducers.DBWriter.SiStripLatencyDummyDBWriter_cfi")

process.SiStripLatencyGenerator.latency = 255
process.SiStripLatencyGenerator.mode = 0
# process.SiStripLatencyGenerator.latency = 143
# process.SiStripLatencyGenerator.mode = 47
# process.SiStripLatencyGenerator.latency = 146
# process.SiStripLatencyGenerator.mode = 37

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(2),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:dbfile.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiStripLatencyRcd'),
        tag = cms.string('SiStripLatency_Ideal_31X')
    ))
)

process.siStripLatencyDummyDBWriter.record=process.PoolDBOutputService.toPut[0].record 

# This specifies the IOV, not the EmptySource (to see why look at the DummyDBWriter code)
# process.siStripLatencyDummyDBWriter.OpenIovAt = ""
# process.siStripLatencyDummyDBWriter.OpenIovAtTime = 1
# process.siStripLatencyDummyDBWriter.OpenIovAtTime = 108449
# process.siStripLatencyDummyDBWriter.OpenIovAtTime = 110177
# process.siStripLatencyDummyDBWriter.OpenIovAtTime = 110213

process.p1 = cms.Path(process.siStripLatencyDummyDBWriter)
