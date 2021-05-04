import FWCore.ParameterSet.Config as cms

process = cms.Process("CONDOBJMON")
#-------------------------------------------------
# CALIBRATION
#-------------------------------------------------
###process.load("DQM.SiStripMonitorSummary.Tags21X_cff")

process.load("DQM.SiStripCommon.TkHistoMap_cff")


#-------------------------------------------------
# DQM
#-------------------------------------------------
process.load("DQM.SiStripMonitorSummary.SiStripMonitorCondData_cfi")
process.load("CalibTracker.Configuration.Tracker_DependentRecords_forGlobalTag_nofakes_cff"
)

process.source = cms.Source("EmptyIOVSource",
                            lastValue = cms.uint64(216000),
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(216000),
                            interval = cms.uint64(1)
                            )


# the DB Geometry is NOT used because in this cfg only one tag is taken from the DB and no GT is used. To be fixed if this is a problem
process.load('Configuration.Geometry.GeometryExtended_cff')
process.TrackerTopologyEP = cms.ESProducer("TrackerTopologyEP")
process.load("Geometry.TrackerGeometryBuilder.trackerParameters_cfi")

process.a = cms.ESSource("PoolDBESSource",
   BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
   DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(2),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('frontier://cmsfrontier.cern.ch:8000/FrontierProd/CMS_COND_31X_FROM21X'),
                         toGet = cms.VPSet(
    cms.PSet(
    record = cms.string('SiStripBadChannelRcd'), 
##    tag = cms.string('SiStripBadChannel_GR09_31X_v1_hlt')
        tag = cms.string('SiStripBadChannel_HotStrip_CRAFT_v3_offline')
##        tag = cms.string('SiStripBadChannel_HotStrip_GR09_31X_v1_hlt')    
    )
##     cms.PSet(
##     record = cms.string('SiStripDetVOffRcd'),
##     tag = cms.string('SiStripDetVOff_Ideal_31X_v2')
##     ),
##     cms.PSet(
##     record = cms.string('SiStripFedCablingRcd'),
##     tag = cms.string('SiStripFedCabling_GR09_31X_v1_hlt')
##     ),
##     cms.PSet(
##     record = cms.string('SiStripBadFiberRcd'),
##     tag = cms.string('SiStripBadFiber_Ideal_31X_v2')
##     ),
##     cms.PSet(
##     record = cms.string('SiStripBadModuleRcd'),
##     tag = cms.string('SiStripBadModule_Ideal_31X_v2')
##     ),
  
    )
)

process.siStripQualityESProducer = cms.ESProducer("SiStripQualityESProducer",
    ThresholdForReducedGranularity = cms.double(0.2),
    appendToDataLabel = cms.string(''),
    ReduceGranularity = cms.bool(False),
    PrintDebugOutput = cms.bool(False),
    UseEmptyRunInfo = cms.bool(False),
    ListOfRecordToMerge = cms.VPSet(cms.PSet(
##    record = cms.string('SiStripDetVOffRcd'),
##    record = cms.string('SiStripDetCablingRcd'),
    record = cms.string('SiStripBadChannelRcd'),        
    tag = cms.string('')
    ))
)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )
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

from DQMServices.Core.DQMQualityTester import DQMQualityTester
process.qTester = DQMQualityTester(
                               qtList = cms.untracked.FileInPath('DQM/SiStripMonitorSummary/data/CondDBQtests.xml'),
                               QualityTestPrescaler = cms.untracked.int32(1),
                               getQualityTestsFromFile = cms.untracked.bool(True)
                               )

process.DQMStore = cms.Service("DQMStore",
                               verbose = cms.untracked.int32(1)
                               )



## --- General Configurable options:

process.CondDataMonitoring.OutputFileName = 'SiStrip_CondDB_CurrentTag.root'

process.CondDataMonitoring.MonitorSiStripPedestal      = False
process.CondDataMonitoring.MonitorSiStripNoise         = False
process.CondDataMonitoring.MonitorSiStripQuality       = True
process.CondDataMonitoring.MonitorSiStripCabling       = False
process.CondDataMonitoring.MonitorSiStripApvGain       = False
process.CondDataMonitoring.MonitorSiStripLorentzAngle  = False
process.CondDataMonitoring.MonitorSiStripBackPlaneCorrection  = False
process.CondDataMonitoring.MonitorSiStripLowThreshold  = False
process.CondDataMonitoring.MonitorSiStripHighThreshold = False

process.CondDataMonitoring.OutputMEsInRootFile         = True
process.CondDataMonitoring.FillConditions_PSet.Mod_On            = False
process.CondDataMonitoring.FillConditions_PSet.OutputSummaryAtLayerLevelAsImage           = False
process.CondDataMonitoring.FillConditions_PSet.OutputSummaryProfileAtLayerLevelAsImage    = False # This should be saved only in case of LA or BP (because for LA and BP no SummaryAtLayerLevel is available)

## --- TkMap specific Configurable options:

process.CondDataMonitoring.SiStripQualityDQM_PSet.TkMap_On     = True
process.CondDataMonitoring.SiStripQualityDQM_PSet.TkMapName     = 'QualityTkMap.svg'
process.CondDataMonitoring.SiStripQualityDQM_PSet.minValue     = 0.
process.CondDataMonitoring.SiStripQualityDQM_PSet.maxValue     = 1.





## ---


process.p = cms.Path(process.CondDataMonitoring*process.qTester)
#process.p = cms.Path(process.CondDataMonitoring)
