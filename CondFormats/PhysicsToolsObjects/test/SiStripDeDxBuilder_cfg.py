# The following comments couldn't be translated into the new config version:

# upload to database 

#string timetype = "timestamp"    

import FWCore.ParameterSet.Config as cms

process = cms.Process("SiStripDeDxMipBuilder")

process.MessageLogger = cms.Service("MessageLogger",
    cablingBuilder = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    debugModules = cms.untracked.vstring(''),
    files = cms.untracked.PSet(
        dedxmipBuilder = cms.untracked.PSet(

        )
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

#process.load("CalibTracker.SiStripESProducers.SiStripFedCablingFakeESSource_cfi")

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(2),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:dummy2.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiStripDeDxMipRcd'),
        tag = cms.string('SiStripDeDxMip_30X')
    ))
)

process.fedcablingbuilder = cms.EDAnalyzer("SiStripDeDxMipBuilder")


process.p1 = cms.Path(process.fedcablingbuilder)


