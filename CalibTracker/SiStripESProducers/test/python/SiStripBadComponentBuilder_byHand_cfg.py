import FWCore.ParameterSet.Config as cms

process = cms.Process("CALIB")

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

#Populate ES
process.load("CalibTracker.SiStripESProducers.fake.SiStripBadModuleConfigurableFakeESSource_cfi")
from CalibTracker.SiStripESProducers.fake.SiStripBadModuleConfigurableFakeESSource_cfi import siStripBadModuleConfigurableFakeESSource
siStripBadModuleConfigurableFakeESSource.BadComponentList = cms.untracked.VPSet(   cms.PSet(
### To mask TOB:
##     SubDet = cms.string('TOB'),
##     layer = cms.uint32(3),    ## SELECTION: layer = 1..6, 0(ALL)	       
##     bkw_frw = cms.uint32(0),  ## bkw_frw = 1(TOB-) 2(TOB+) 0(everything)     
##     rod = cms.uint32(0),      ## rod = 1..N, 0(ALL)			       
##     detid = cms.uint32(0),    ## detid number = 0 (ALL),  specific number 
##     ster = cms.uint32(0)      ## ster = 1(stereo), 2 (nonstereo), 0(ALL)

### To mask TIB:
##     SubDet = cms.string('TIB'),  
##     layer = cms.uint32(2),      ## SELECTION: layer = 1..4, 0(ALL)		    
##     bkw_frw = cms.uint32(0),    ## bkw_frw = 1(TIB-), 2(TIB+) 0(ALL)	    
##     int_ext = cms.uint32(0),    ## int_ext = 1 (internal), 2(external), 0(ALL)  
##     ster = cms.uint32(0),       ## ster = 1(stereo), 2 (nonstereo), 0(ALL)	    
##     string_ = cms.uint32(0),    ## string = 1..N, 0(ALL)			    
##     detid = cms.uint32(0)       ## detid number = 0 (ALL),  specific number 

### To mask TID:
##     SubDet = cms.string('TID'), 
##     side = cms.uint32(0),       ## SELECTION: side = 1(back, Z-), 2(front, Z+), 0(ALL)	 
##     wheel = cms.uint32(2),      ## wheel = 1..3, 0(ALL)					 
##     ring = cms.uint32(0),       ## ring  = 1..3, 0(ALL)					 
##     ster = cms.uint32(0),       ## ster = 1(stereo), 2 (nonstereo), 0(ALL)		 
##     detid = cms.uint32(0)       ## detid number = 0 (ALL),  specific number

### To mask TEC:
    SubDet = cms.string('TEC'), 
    side = cms.uint32(1),          ## SELECTION: side = 1(back, Z-), 2(front, Z+), 0(ALL)	 
    wheel = cms.uint32(0),         ## wheel = 1..9, 0(ALL)					 
    ring = cms.uint32(0),          ## ring  = 1..7, 0(ALL)
    petal_bkw_frw = cms.uint32(0), ## petal_bkw_frw = 1(backward) 2(forward) 0(all)
    petal = cms.uint32(7),         ## petal = 1..8, 0(ALL)
    ster = cms.uint32(0),          ## ster = 1(stereo), 2 (nonstereo), 0(ALL)		 
    detid = cms.uint32(0),         ## detid number = 0 (ALL),  specific number

    # This list is independent on all the rest. Any module appearing here will be masked.
    detidList = cms.untracked.vuint32(402666125,
                            402666126,
                            402666129,
                            402666130,
                            402666133,
                            402666134,
                            402666137,
                            402666138,
                            402666141,
                            402666142,
                            402666145,
                            402666146,
                            402666257,
                            402666258,
                            402666261,
                            402666262,
                            402666265,
                            402666266,
                            402666269,
                            402666270,
                            402666273,
                            402666274,
                            402666277,
                            402666278,
                            402666629,
                            402666630,
                            402666633,
                            402666634,
                            402666637,
                            402666638,
                            402666641,
                            402666642,
                            402666669,
                            402666670,
                            402666673,
                            402666674,
                            402666777,
                            402666778,
                            402666781,
                            402666782,
                            402666785,
                            402666786,
                            402666789,
                            402666790,
                            402666793,
                            402666794,
                            402666797,
                            402666798
                            )
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
        tag = cms.string('SiStripBadComponents_TECminus')
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


