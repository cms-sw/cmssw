import FWCore.ParameterSet.Config as cms

process = cms.Process("CALIB")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring(''),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    destinations = cms.untracked.vstring('cout')
)

process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1), # Specify the first run number for which the masking should be done
    lastValue = cms.uint64(1),  # Specify the first run number for which the masking should be done
    interval = cms.uint64(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)


#DetId369141286_APV0
#DetId369141286_APV1
#DetId369141862_APV0
#DetId369141862_APV1
#DetId369157180_APV0
#DetId369157180_APV1
#DetId436244845_APV0
#DetId436244845_APV1
#DetId436245994_APV0
#DetId436245994_APV1
#DetId436248977_APV0

#Populate ES
process.SiStripDetInfoFileReader = cms.Service("SiStripDetInfoFileReader")
process.load("CalibTracker.SiStripESProducers.fake.SiStripBadModuleConfigurableFakeESSource_cfi")
from CalibTracker.SiStripESProducers.fake.SiStripBadModuleConfigurableFakeESSource_cfi import siStripBadModuleConfigurableFakeESSource
siStripBadModuleConfigurableFakeESSource.doByAPVs = cms.untracked.bool(True)  
siStripBadModuleConfigurableFakeESSource.BadComponentList = cms.untracked.VPSet()
siStripBadModuleConfigurableFakeESSource.BadAPVList = cms.untracked.VPSet(
    cms.PSet(
        DetId = cms.uint32(369141286),        	 
        APVs = cms.vuint32(0,1),       
        ),
    cms.PSet(
        DetId = cms.uint32(369141862),        	 
        APVs = cms.vuint32(0,1),       
        ),
    cms.PSet(
        DetId = cms.uint32(369157180),        	 
        APVs = cms.vuint32(0,1),       
        ),
    cms.PSet(
        DetId = cms.uint32(436244845),        	 
        APVs = cms.vuint32(0,1),       
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
        tag = cms.string('SiStripBadComponentsToMask')
        ))
)


###### Quality ESProducer                                                                       
process.load("CalibTracker.SiStripESProducers.SiStripQualityESProducer_cfi")
process.siStripQualityESProducer.ListOfRecordToMerge = cms.VPSet(
     cms.PSet( record = cms.string("SiStripBadModuleRcd"),  tag    = cms.string("") )
     )

#### Add these lines to produce a tracker map
process.load("DQM.SiStripCommon.TkHistoMap_cff")
# load TrackerTopology (needed for TkDetMap and TkHistoMap)
process.load("Geometry.CMSCommonData.cmsExtendedGeometry2017XML_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerParameters_cfi")
process.TrackerTopologyEP = cms.ESProducer("TrackerTopologyEP")
####

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
process.reader = DQMEDAnalyzer("SiStripQualityStatistics",
                               dataLabel = cms.untracked.string(""),
                               TkMapFileName = cms.untracked.string("TkMapBadComponents_byHand.png")
                               )

process.siStripBadModuleDummyDBWriter.record=process.PoolDBOutputService.toPut[0].record
process.p = cms.Path(process.reader*process.siStripBadModuleDummyDBWriter)


