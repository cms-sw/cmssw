import FWCore.ParameterSet.Config as cms

import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("Reader")

options = VarParsing.VarParsing("analysis")

options.register ('logDestination',
                  "",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "log file")
options.register ('qualityLogDestination',
                  "",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "quality log file")
options.register ('runInfoTag',
                  "",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "RunInfo tag name")
options.register ('cablingTag',
                  "",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "cabling tag name")
options.register ('cablingConnectionString',
                  "",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "Cabling connection string")
options.register ('runinfoConnectionString',
                  "",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "RunInfo connection string")
options.register ('MonitorCumulative',
                  False,
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.bool,          # string, int, or float
                  "Cumulative Monitoring?")
options.register ('outputRootFile',
                  "",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "output root file")
options.register ('runNumber',
                  0,
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "run number")

options.parseArguments()

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring(''),
#    insertLog = cms.untracked.PSet(
#        threshold = cms.untracked.string('INFO')
#    ),
                                    destinations = cms.untracked.vstring(options.logDestination,
                                                                         options.qualityLogDestination
#                                                                         options.cablingLogDestination,
#                                                                         options.condLogDestination
                                                                         ), #Reader, cout
                                    categories = cms.untracked.vstring('SiStripQualityStatistics'
#                                                                       'SiStripQualityDQM',
#                                                                       'SiStripFedCablingReader',
#                                                                       'DummyCondObjContentPrinter',
                                                                       )
)
setattr(process.MessageLogger,options.logDestination,cms.untracked.PSet(threshold = cms.untracked.string('INFO')))
setattr(process.MessageLogger,options.qualityLogDestination,cms.untracked.PSet(
    threshold = cms.untracked.string('INFO'),
    default = cms.untracked.PSet(limit=cms.untracked.int32(0)),
    SiStripQualityStatistics = cms.untracked.PSet(limit=cms.untracked.int32(100000))
#    SiStripQualityDQM = cms.untracked.PSet(limit=cms.untracked.int32(100000))
    )
        )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("EmptyIOVSource",
    firstValue = cms.uint64(options.runNumber),
    lastValue = cms.uint64(options.runNumber),
    timetype = cms.string('runnumber'),
    interval = cms.uint64(1)
)

process.poolDBESSourceRunInfo = cms.ESSource("PoolDBESSource",
   BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
   DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(1),  # it used to be 2
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string(options.runinfoConnectionString),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('RunInfoRcd'),
        tag = cms.string(options.runInfoTag)
        )               
                      )
)

process.poolDBESSourceFedCabling = cms.ESSource("PoolDBESSource",
   BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
   DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(1),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string(options.cablingConnectionString),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('SiStripFedCablingRcd'),
        tag = cms.string(options.cablingTag)
        )               
                      )
)

process.DQMStore = cms.Service("DQMStore",
    referenceFileName = cms.untracked.string(''),
    verbose = cms.untracked.int32(1)
)

process.load("DQM.SiStripMonitorSummary.SiStripMonitorCondData_cfi")

process.CondDataMonitoring.OutputFileName = options.outputRootFile
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
process.CondDataMonitoring.FillConditions_PSet.OutputSummaryAtLayerLevelAsImage           = True
process.CondDataMonitoring.FillConditions_PSet.OutputSummaryProfileAtLayerLevelAsImage    = False # This should be saved only in case of LA (because for LA no SummaryAtLayerLevel is available)
process.CondDataMonitoring.FillConditions_PSet.OutputCumulativeSummaryAtLayerLevelAsImage = options.MonitorCumulative
process.CondDataMonitoring.FillConditions_PSet.HistoMaps_On     = False
process.CondDataMonitoring.FillConditions_PSet.TkMap_On         = True # This is just for test until TkMap is included in all classes!!! Uncomment!!!!
process.CondDataMonitoring.FillConditions_PSet.ActiveDetIds_On  = True # This should be set to False only for Lorentz Angle

process.CondDataMonitoring.SiStripPedestalsDQM_PSet.FillSummaryAtLayerLevel     = True
process.CondDataMonitoring.SiStripNoisesDQM_PSet.FillSummaryAtLayerLevel        = True
process.CondDataMonitoring.SiStripQualityDQM_PSet.FillSummaryAtLayerLevel       = True
process.CondDataMonitoring.SiStripApvGainsDQM_PSet.FillSummaryAtLayerLevel      = True
process.CondDataMonitoring.SiStripLowThresholdDQM_PSet.FillSummaryAtLayerLevel  = True
process.CondDataMonitoring.SiStripHighThresholdDQM_PSet.FillSummaryAtLayerLevel = True

process.CondDataMonitoring.SiStripCablingDQM_PSet.CondObj_fillId       = 'ProfileAndCumul'
process.CondDataMonitoring.SiStripPedestalsDQM_PSet.CondObj_fillId     = 'onlyProfile'
process.CondDataMonitoring.SiStripNoisesDQM_PSet.CondObj_fillId        = 'onlyCumul'
process.CondDataMonitoring.SiStripQualityDQM_PSet.CondObj_fillId       = 'onlyProfile'
process.CondDataMonitoring.SiStripApvGainsDQM_PSet.CondObj_fillId      = 'ProfileAndCumul'
process.CondDataMonitoring.SiStripLorentzAngleDQM_PSet.CondObj_fillId  = 'ProfileAndCumul'
process.CondDataMonitoring.SiStripBackPlaneCorrectionDQM_PSet.CondObj_fillId  = 'ProfileAndCumul'
process.CondDataMonitoring.SiStripLowThresholdDQM_PSet.CondObj_fillId  = 'onlyProfile'
process.CondDataMonitoring.SiStripHighThresholdDQM_PSet.CondObj_fillId = 'onlyProfile'

## --- TkMap specific Configurable options:

process.CondDataMonitoring.SiStripQualityDQM_PSet.TkMap_On     = True
process.CondDataMonitoring.SiStripQualityDQM_PSet.TkMapName    = 'QualityTkMap.png'
process.CondDataMonitoring.SiStripQualityDQM_PSet.minValue     = 0.
process.CondDataMonitoring.SiStripQualityDQM_PSet.maxValue     = 100.

process.CondDataMonitoring.SiStripCablingDQM_PSet.TkMap_On     = True
process.CondDataMonitoring.SiStripCablingDQM_PSet.TkMapName    = 'CablingTkMap.png'
process.CondDataMonitoring.SiStripCablingDQM_PSet.minValue     = 0.
process.CondDataMonitoring.SiStripCablingDQM_PSet.maxValue     = 6.

process.CondDataMonitoring.SiStripPedestalsDQM_PSet.TkMap_On     = True
process.CondDataMonitoring.SiStripPedestalsDQM_PSet.TkMapName    = 'PedestalTkMap.png'
process.CondDataMonitoring.SiStripPedestalsDQM_PSet.minValue     = 0.
process.CondDataMonitoring.SiStripPedestalsDQM_PSet.maxValue     = 400.

process.CondDataMonitoring.SiStripNoisesDQM_PSet.TkMap_On     = True
process.CondDataMonitoring.SiStripNoisesDQM_PSet.TkMapName    = 'NoiseTkMap.png'
process.CondDataMonitoring.SiStripNoisesDQM_PSet.minValue     = 3.
process.CondDataMonitoring.SiStripNoisesDQM_PSet.maxValue     = 9.

process.CondDataMonitoring.SiStripApvGainsDQM_PSet.TkMap_On     = True
process.CondDataMonitoring.SiStripApvGainsDQM_PSet.TkMapName    = 'GainTkMap.png'
process.CondDataMonitoring.SiStripApvGainsDQM_PSet.minValue     = 0.
process.CondDataMonitoring.SiStripApvGainsDQM_PSet.maxValue     = 1.5

process.CondDataMonitoring.SiStripLorentzAngleDQM_PSet.TkMap_On     = True
process.CondDataMonitoring.SiStripLorentzAngleDQM_PSet.TkMapName    = 'LorentzAngleTkMap.png'
process.CondDataMonitoring.SiStripLorentzAngleDQM_PSet.minValue     = 0.01
process.CondDataMonitoring.SiStripLorentzAngleDQM_PSet.maxValue     = 0.03

process.CondDataMonitoring.SiStripBackPlaneCorrectionDQM_PSet.TkMap_On     = True
process.CondDataMonitoring.SiStripBackPlaneCorrectionDQM_PSet.TkMapName    = 'BackPlaneCorrectionTkMap.png'
process.CondDataMonitoring.SiStripBackPlaneCorrectionDQM_PSet.minValue     = 0.00
process.CondDataMonitoring.SiStripBackPlaneCorrectionDQM_PSet.maxValue     = 0.10

process.CondDataMonitoring.SiStripLowThresholdDQM_PSet.TkMap_On     = True
process.CondDataMonitoring.SiStripLowThresholdDQM_PSet.TkMapName     = 'LowThresholdTkMap.png'
process.CondDataMonitoring.SiStripLowThresholdDQM_PSet.minValue     = 0.
process.CondDataMonitoring.SiStripLowThresholdDQM_PSet.maxValue     = 10.

process.CondDataMonitoring.SiStripHighThresholdDQM_PSet.TkMap_On     = True
process.CondDataMonitoring.SiStripHighThresholdDQM_PSet.TkMapName     = 'HighThresholdTkMap.png'
process.CondDataMonitoring.SiStripHighThresholdDQM_PSet.minValue     = 0.
process.CondDataMonitoring.SiStripHighThresholdDQM_PSet.maxValue     = 10.


process.p1 = cms.Path(process.CondDataMonitoring)

process.SiStripQualityESProducer = cms.ESProducer("SiStripQualityESProducer",
   ReduceGranularity = cms.bool(False),
   PrintDebugOutput = cms.bool(False),
   UseEmptyRunInfo = cms.bool(True),
   ListOfRecordToMerge = cms.VPSet(
    cms.PSet(
    record = cms.string('RunInfoRcd'),
    tag = cms.string('')
    ),
    cms.PSet(
    record = cms.string('SiStripDetCablingRcd'),
    tag = cms.string('')
    )
                                   )
)

process.sistripconn = cms.ESProducer("SiStripConnectivity")

process.stat = cms.EDAnalyzer("SiStripQualityStatistics",
    TkMapFileName = cms.untracked.string(''),
    dataLabel = cms.untracked.string('')
)

process.e = cms.EndPath(process.stat)
