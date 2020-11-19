import FWCore.ParameterSet.Config as cms

process = cms.Process("CALIB")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    debugModules = cms.untracked.vstring(''),
    files = cms.untracked.PSet(
        QualityReader = cms.untracked.PSet(
            threshold = cms.untracked.string('INFO')
        )
    )
)

process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(10),
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(10),
    interval = cms.uint64(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

#Populate ES
process.load("CalibTracker.SiStripESProducers.fake.SiStripBadModuleConfigurableFakeESSource_cfi")
from CalibTracker.SiStripESProducers.fake.SiStripBadModuleConfigurableFakeESSource_cfi import siStripBadModuleConfigurableFakeESSource
siStripBadModuleConfigurableFakeESSource.BadComponentList = cms.untracked.VPSet(   cms.PSet(
    SubDet = cms.string('TID'), 
    wheel = cms.uint32(1),      ## SELECTION: side = 1(back, Z-), 2(front, Z+), 0(ALL)	 
    detid = cms.uint32(0),      ## wheel = 1..3, 0(ALL)					 
    ster = cms.uint32(0),       ## ring  = 1..3, 0(ALL)					 
    ring = cms.uint32(0),       ## ster = 1(stereo), 2 (nonstereo), 0(ALL)		 
    side = cms.uint32(0)            ## detid number = 0 (ALL),  specific number           
    )
)

#Write on DB
process.load("CalibTracker.SiStripESProducers.DBWriter.SiStripBadModuleDummyDBWriter_cfi")
process.siStripBadModuleDummyDBWriter.OpenIovAt = cms.untracked.string("currentTime")

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(2),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:dbfile.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiStripBadStrip'),
        tag = cms.string('SiStripBadModule_Fake_TID')
        ))
)


###### Quality ESProducer                                                                       
process.load("CalibTracker.SiStripESProducers.SiStripQualityESProducer_cfi")
process.siStripQualityESProducer.ListOfRecordToMerge = cms.VPSet(
     cms.PSet( record = cms.string("SiStripBadModuleRcd"),  tag    = cms.string("") )
     )

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
process.reader = DQMEDAnalyzer("SiStripQualityStatistics",
                               dataLabel = cms.untracked.string(""),
                               TkMapFileName = cms.untracked.string("")
                               )

process.siStripBadModuleDummyDBWriter.record=process.PoolDBOutputService.toPut[0].record
process.p = cms.Path(process.reader*process.siStripBadModuleDummyDBWriter)


