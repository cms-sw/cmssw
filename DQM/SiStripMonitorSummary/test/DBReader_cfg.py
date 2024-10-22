import FWCore.ParameterSet.Config as cms

process = cms.Process("Reader")

process.load("DQM.SiStripCommon.TkHistoMap_cff")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('INFO')
    ),
    debugModules = cms.untracked.vstring(''),
    files = cms.untracked.PSet(
        Reader = cms.untracked.PSet(
            threshold = cms.untracked.string('INFO')
        )
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("EmptyIOVSource",
    firstValue = cms.uint64(70873),
    lastValue = cms.uint64(70873),
    timetype = cms.string('runnumber'),
    interval = cms.uint64(1)
)

# the DB Geometry is NOT used because in this cfg only one tag is taken from the DB and no GT is used. To be fixed if this is a problem
process.load('Configuration.Geometry.GeometryExtended_cff')
process.TrackerTopologyEP = cms.ESProducer("TrackerTopologyEP")
process.load("Geometry.TrackerGeometryBuilder.trackerParameters_cfi")

process.poolDBESSource = cms.ESSource("PoolDBESSource",
   BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
   DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(2),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('frontier://cmsfrontier.cern.ch:8000/FrontierProd/CMS_COND_31X_FROM21X'),
    toGet = cms.VPSet(
    cms.PSet(
        record = cms.string('SiStripPedestalsRcd'),
        tag = cms.string('SiStripPedestals_CRAFT_21X_v4_offline')
    ),
    cms.PSet(
        record = cms.string('SiStripNoisesRcd'),
        tag = cms.string('SiStripNoise_CRAFT_21X_v4_offline')
    ),
##     cms.PSet(
##         record = cms.string('SiStripQualityRcd'), 
##         tag = cms.string('SiStripBadChannel_CRAFT_21X_v4_offline') 
##     ),
    cms.PSet(
        record = cms.string('SiStripApvGainRcd'),
        tag = cms.string('SiStripGain_CRAFT_22X_v1_offline') 
    ),
    cms.PSet(
         record = cms.string('SiStripLorentzAngleRcd'),
         tag = cms.string('SiStripLorentzAngle_CRAFT_22X_v1_offline') 
     )
    
    )
)


process.a = cms.ESSource("PoolDBESSource",
   BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
   DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(2),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('frontier://cmsfrontier.cern.ch:8000/FrontierProd/CMS_COND_31X_STRIP'),
    toGet = cms.VPSet(
    cms.PSet(
        record = cms.string('SiStripFedCablingRcd'),
        tag = cms.string('SiStripFedCabling_CRAFT_31X_v1_offline')
    ),
    cms.PSet(
        record = cms.string('SiStripBadStripRcd'), 
        tag = cms.string('SiStripBadComponents_realisticMC_31X_v1') 
    ),
     cms.PSet(
         record = cms.string('SiStripThresholdRcd'),
         tag = cms.string('SiStripThreshold_CRAFT_31X_v1_offline')
     )
   
    )
)


siStripQualityESProducer = cms.ESProducer("SiStripQualityESProducer",
    ThresholdForReducedGranularity = cms.double(0.2),
    appendToDataLabel = cms.string(''),
    ReduceGranularity = cms.bool(True),
    ListOfRecordToMerge = cms.VPSet(cms.PSet(
        record = cms.string('SiStripDetVOffRcd"'),
##        record = cms.string('SiStripDetCablingRcd'),
##        record = cms.string('SiStripBadChannelRcd'),        
        tag = cms.string('test')
    ))
)


process.DQMStore = cms.Service("DQMStore",
    verbose = cms.untracked.int32(1)
)

process.load("DQM.SiStripMonitorSummary.SiStripMonitorCondData_cfi")
process.load("CalibTracker.Configuration.Tracker_DependentRecords_forGlobalTag_nofakes_cff"
)


## --- General Configurable options:

process.CondDataMonitoring.OutputFileName = 'SiStrip_CondDB_CurrentTag.root'

process.CondDataMonitoring.MonitorSiStripPedestal      = False
process.CondDataMonitoring.MonitorSiStripNoise         = False
process.CondDataMonitoring.MonitorSiStripQuality       = False
process.CondDataMonitoring.MonitorSiStripCabling       = False
process.CondDataMonitoring.MonitorSiStripApvGain       = False
process.CondDataMonitoring.MonitorSiStripLorentzAngle  = True
process.CondDataMonitoring.MonitorSiStripBackPlaneCorrection  = False
process.CondDataMonitoring.MonitorSiStripLowThreshold  = False
process.CondDataMonitoring.MonitorSiStripHighThreshold = False

process.CondDataMonitoring.OutputMEsInRootFile         = True
process.CondDataMonitoring.FillConditions_PSet.OutputSummaryAtLayerLevelAsImage           = False
process.CondDataMonitoring.FillConditions_PSet.OutputSummaryProfileAtLayerLevelAsImage    = False # This should be saved only in case of LA (because for LA no SummaryAtLayerLevel is available)

## --- TkMap specific Configurable options:

process.CondDataMonitoring.SiStripCablingDQM_PSet.TkMap_On     = True
process.CondDataMonitoring.SiStripCablingDQM_PSet.TkMapName     = 'CablingTkMap.png'
process.CondDataMonitoring.SiStripCablingDQM_PSet.minValue     = 0.
process.CondDataMonitoring.SiStripCablingDQM_PSet.maxValue     = 6.

process.CondDataMonitoring.SiStripPedestalsDQM_PSet.TkMap_On     = True
process.CondDataMonitoring.SiStripPedestalsDQM_PSet.TkMapName     = 'PedestalTkMap.svg'
process.CondDataMonitoring.SiStripPedestalsDQM_PSet.minValue     = 0.
process.CondDataMonitoring.SiStripPedestalsDQM_PSet.maxValue     = 400.

process.CondDataMonitoring.SiStripNoisesDQM_PSet.TkMap_On     = True
process.CondDataMonitoring.SiStripNoisesDQM_PSet.TkMapName     = 'NoiseTkMap.svg'
process.CondDataMonitoring.SiStripNoisesDQM_PSet.minValue     = 0.
process.CondDataMonitoring.SiStripNoisesDQM_PSet.maxValue     = 6.

process.CondDataMonitoring.SiStripApvGainsDQM_PSet.TkMap_On     = True
process.CondDataMonitoring.SiStripApvGainsDQM_PSet.TkMapName     = 'GainTkMap.svg'
process.CondDataMonitoring.SiStripApvGainsDQM_PSet.minValue     = 0.
process.CondDataMonitoring.SiStripApvGainsDQM_PSet.maxValue     = 1.5

process.CondDataMonitoring.SiStripLorentzAngleDQM_PSet.TkMap_On     = True
process.CondDataMonitoring.SiStripLorentzAngleDQM_PSet.TkMapName     = 'LorentzAngleTkMap.png'
process.CondDataMonitoring.SiStripLorentzAngleDQM_PSet.minValue     = 0.01
process.CondDataMonitoring.SiStripLorentzAngleDQM_PSet.maxValue     = 0.03

process.CondDataMonitoring.SiStripBackPlaneCorrectionDQM_PSet.TkMap_On     = True
process.CondDataMonitoring.SiStripBackPlaneCorrectionDQM_PSet.TkMapName     = 'BackPlaneCorrectionTkMap.png'
process.CondDataMonitoring.SiStripBackPlaneCorrectionDQM_PSet.minValue     = 0.00
process.CondDataMonitoring.SiStripBackPlaneCorrectionDQM_PSet.maxValue     = 0.10

process.CondDataMonitoring.SiStripLowThresholdDQM_PSet.TkMap_On     = True
process.CondDataMonitoring.SiStripLowThresholdDQM_PSet.TkMapName     = 'LowThresholdTkMap.svg'
process.CondDataMonitoring.SiStripLowThresholdDQM_PSet.minValue     = 0.
process.CondDataMonitoring.SiStripLowThresholdDQM_PSet.maxValue     = 10.

process.CondDataMonitoring.SiStripHighThresholdDQM_PSet.TkMap_On     = True
process.CondDataMonitoring.SiStripHighThresholdDQM_PSet.TkMapName     = 'HighThresholdTkMap.svg'
process.CondDataMonitoring.SiStripHighThresholdDQM_PSet.minValue     = 0.
process.CondDataMonitoring.SiStripHighThresholdDQM_PSet.maxValue     = 10.

## ---
process.p1 = cms.Path(process.CondDataMonitoring)
