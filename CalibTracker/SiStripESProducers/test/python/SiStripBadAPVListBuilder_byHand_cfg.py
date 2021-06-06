import FWCore.ParameterSet.Config as cms
import six
process = cms.Process("CALIB")

####################################################
def getFileInPath(rfile):
####################################################
   import os
   for dir in os.environ['CMSSW_SEARCH_PATH'].split(":"):
     if os.path.exists(os.path.join(dir,rfile)): return os.path.join(dir,rfile)
   return None

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('INFO')
    ),
    debugModules = cms.untracked.vstring('')
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


### Example of APVs to be masked
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

detIDsFileName = getFileInPath('CalibTracker/SiStripCommon/data/SiStripDetInfo.dat')
detDict = {}
with open(detIDsFileName,"r") as detIDs:  # create dictionary online -> rawid
    for entry in detIDs:
        fields = entry.strip().split()
        detDict[fields[0]]=fields[1]

#print(detDict)

APVsToKill = []
for det,napv in six.iteritems(detDict):
    APVsToKill.append(
        cms.PSet(
            DetId = cms.uint32(int(det)),        	 
            APVs = cms.vuint32( 0,1 if int(napv)<6 else 2,3,4,5  ),       
            )
        )

#Populate ES
process.load("CalibTracker.SiStripESProducers.fake.SiStripBadModuleConfigurableFakeESSource_cfi")
from CalibTracker.SiStripESProducers.fake.SiStripBadModuleConfigurableFakeESSource_cfi import siStripBadModuleConfigurableFakeESSource
siStripBadModuleConfigurableFakeESSource.doByAPVs = cms.untracked.bool(True)  
siStripBadModuleConfigurableFakeESSource.BadComponentList = cms.untracked.VPSet()
siStripBadModuleConfigurableFakeESSource.BadAPVList = cms.untracked.VPSet(*APVsToKill)
    # cms.PSet(
    #     DetId = cms.uint32(369141286),        	 
    #     APVs = cms.vuint32(0,1),       
    #     ),
    # cms.PSet(
    #     DetId = cms.uint32(369141862),        	 
    #     APVs = cms.vuint32(0,1),       
    #     ),
    # cms.PSet(
    #     DetId = cms.uint32(369157180),        	 
    #     APVs = cms.vuint32(0,1),       
    #     ),
    # cms.PSet(
    #     DetId = cms.uint32(436244845),        	 
    #     APVs = cms.vuint32(0,1),       
    #     )
    #)

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
# load TrackerTopology (needed for TkDetMap and TkHistoMap)
process.load("Configuration.Geometry.GeometryExtended2017_cff")
process.load("Geometry.TrackerGeometryBuilder.trackerParameters_cfi")
process.TrackerTopologyEP = cms.ESProducer("TrackerTopologyEP")
process.load("DQM.SiStripCommon.TkHistoMap_cff")
####

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
process.reader = DQMEDAnalyzer("SiStripQualityStatistics",
                               dataLabel = cms.untracked.string(""),
                               TkMapFileName = cms.untracked.string("TkMapBadComponents_byHand.png")
                               )

process.siStripBadModuleDummyDBWriter.record=process.PoolDBOutputService.toPut[0].record
process.p = cms.Path(process.reader*process.siStripBadModuleDummyDBWriter)


