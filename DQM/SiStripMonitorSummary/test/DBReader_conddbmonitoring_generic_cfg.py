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
options.register ('cablingLogDestination',
                  "",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "cabling log file")
options.register ('condLogDestination',
                  "",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "conditions log file")
options.register ('outputRootFile',
                  "",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "output root file")
options.register ('connectionString',
                  "",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "connection string")
options.register ('recordName',
                  "",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "record name")
options.register ('recordForQualityName',
                  "SiStripDetCablingRcd",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "record name")
options.register ('tagName',
                  "",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "tag name")
options.register ('runNumber',
                  0,
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "run number")
options.register ('LatencyMon',
                  False,
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.bool,          # string, int, or float
                  "Monitor latency?")
options.register ('ALCARecoTriggerBitsMon',
                  False,
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.bool,          # string, int, or float
                  "Monitor ALCAReco trigger bits")
options.register ('ShiftAndCrosstalkMon',
                  False,
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.bool,          # string, int, or float
                  "Monitor shift and crosstalk?")
options.register ('PedestalMon',
                  False,
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.bool,          # string, int, or float
                  "Monitor pedestals?")
options.register ('NoiseMon',
                  False,
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.bool,          # string, int, or float
                  "Monitor noise?")
options.register ('QualityMon',
                  False,
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.bool,          # string, int, or float
                  "Monitor quality?")
options.register ('CablingMon',
                  False,
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.bool,          # string, int, or float
                  "Monitor cabling?")
options.register ('GainMon',
                  False,
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.bool,          # string, int, or float
                  "Monitor gain?")
options.register ('LorentzAngleMon',
                  False,
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.bool,          # string, int, or float
                  "Monitor LA?")
options.register ('ThresholdMon',
                  False,
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.bool,          # string, int, or float
                  "Monitor thresholds?")
options.register ('MonitorCumulative',
                  False,
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.bool,          # string, int, or float
                  "Cumulative Monitoring?")
options.register ('ActiveDetId',
                  False,
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.bool,          # string, int, or float
                  "Active detid?")

options.parseArguments()


process.MessageLogger = cms.Service("MessageLogger",
                                    debugModules = cms.untracked.vstring(''),
#    cout = cms.untracked.PSet(
#        threshold = cms.untracked.string('INFO')
                                    destinations = cms.untracked.vstring(options.logDestination,
                                                                         options.qualityLogDestination,
                                                                         options.cablingLogDestination,
                                                                         options.condLogDestination
                                                                         ), #Reader, cout
                                    categories = cms.untracked.vstring('SiStripQualityStatistics',
                                                                       'SiStripQualityDQM',
                                                                       'SiStripFedCablingReader',
                                                                       'DummyCondObjContentPrinter',
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
setattr(process.MessageLogger,options.cablingLogDestination,cms.untracked.PSet(
    threshold = cms.untracked.string('INFO'),
    default = cms.untracked.PSet(limit=cms.untracked.int32(0)),
    SiStripFedCablingReader = cms.untracked.PSet(limit=cms.untracked.int32(100000))
    )
        )
setattr(process.MessageLogger,options.condLogDestination,cms.untracked.PSet(
    threshold = cms.untracked.string('INFO'),
    default = cms.untracked.PSet(limit=cms.untracked.int32(0)),
    DummyCondObjContentPrinter = cms.untracked.PSet(limit=cms.untracked.int32(100000))
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

#process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load('Configuration.Geometry.GeometryExtended_cff')
process.TrackerTopologyEP = cms.ESProducer("TrackerTopologyEP")

process.poolDBESSource = cms.ESSource("PoolDBESSource",
   BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
   DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(1),  # it used to be 2
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string(options.connectionString),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string(options.recordName),
        tag = cms.string(options.tagName)
    ))
)

if options.LatencyMon == True:
    process.reader = cms.EDAnalyzer("SiStripLatencyDummyPrinter")
    process.p1 = cms.Path(process.reader)

elif options.ShiftAndCrosstalkMon == True:
    process.reader = cms.EDAnalyzer("SiStripConfObjectDummyPrinter")
    process.p1 = cms.Path(process.reader)

elif options.ALCARecoTriggerBitsMon == True:
    process.AlCaRecoTriggerBitsRcdRead = cms.EDAnalyzer( "AlCaRecoTriggerBitsRcdRead"
                                                         , outputType  = cms.untracked.string( 'text' )
                                                         , rawFileName = cms.untracked.string( 'AlCaRecoTriggerBitsInfo_RuninsertRun' )
                                                         )
    process.p = cms.Path(process.AlCaRecoTriggerBitsRcdRead)

else:

    process.DQMStore = cms.Service("DQMStore",
                                   referenceFileName = cms.untracked.string(''),
                                   verbose = cms.untracked.int32(1)
                                   )

    process.load("DQM.SiStripMonitorSummary.SiStripMonitorCondData_cfi")

    process.CondDataMonitoring.OutputFileName = options.outputRootFile
    process.CondDataMonitoring.MonitorSiStripPedestal      = options.PedestalMon
    process.CondDataMonitoring.MonitorSiStripNoise         = options.NoiseMon
    process.CondDataMonitoring.MonitorSiStripQuality       = options.QualityMon
    process.CondDataMonitoring.MonitorSiStripCabling       = options.CablingMon
    process.CondDataMonitoring.MonitorSiStripApvGain       = options.GainMon
    process.CondDataMonitoring.MonitorSiStripLorentzAngle  = options.LorentzAngleMon
    process.CondDataMonitoring.MonitorSiStripLowThreshold  = options.ThresholdMon
    process.CondDataMonitoring.MonitorSiStripHighThreshold = options.ThresholdMon
    process.CondDataMonitoring.OutputMEsInRootFile         = True
    process.CondDataMonitoring.FillConditions_PSet.OutputSummaryAtLayerLevelAsImage           = True
    process.CondDataMonitoring.FillConditions_PSet.OutputSummaryProfileAtLayerLevelAsImage    = options.LorentzAngleMon # This should be saved only in case of LA (because for LA no SummaryAtLayerLevel is available)
    process.CondDataMonitoring.FillConditions_PSet.OutputCumulativeSummaryAtLayerLevelAsImage = options.MonitorCumulative
    process.CondDataMonitoring.FillConditions_PSet.HistoMaps_On     = False
    process.CondDataMonitoring.FillConditions_PSet.TkMap_On         = True # This is just for test until TkMap is included in all classes!!! Uncomment!!!!
    process.CondDataMonitoring.FillConditions_PSet.ActiveDetIds_On  = options.ActiveDetId # This should be set to False only for Lorentz Angle
    process.CondDataMonitoring.FillConditions_PSet.Mod_On           = False # Set to True if you want to have single module histograms
    
    process.CondDataMonitoring.SiStripPedestalsDQM_PSet.FillSummaryAtLayerLevel     = True
    process.CondDataMonitoring.SiStripNoisesDQM_PSet.FillSummaryAtLayerLevel        = True
    process.CondDataMonitoring.SiStripQualityDQM_PSet.FillSummaryAtLayerLevel       = True
    process.CondDataMonitoring.SiStripApvGainsDQM_PSet.FillSummaryAtLayerLevel      = True
    process.CondDataMonitoring.SiStripLowThresholdDQM_PSet.FillSummaryAtLayerLevel  = True
    process.CondDataMonitoring.SiStripHighThresholdDQM_PSet.FillSummaryAtLayerLevel = True
    
    process.CondDataMonitoring.SiStripCablingDQM_PSet.CondObj_fillId       = 'ProfileAndCumul'
    process.CondDataMonitoring.SiStripPedestalsDQM_PSet.CondObj_fillId     = 'onlyProfile'    # Use 'ProfileAndCumul' if you want to have single module histograms
    process.CondDataMonitoring.SiStripNoisesDQM_PSet.CondObj_fillId        = 'onlyCumul'      # Use 'ProfileAndCumul' if you want to have single module histograms
    process.CondDataMonitoring.SiStripQualityDQM_PSet.CondObj_fillId       = 'onlyProfile'
    process.CondDataMonitoring.SiStripApvGainsDQM_PSet.CondObj_fillId      = 'ProfileAndCumul'
    process.CondDataMonitoring.SiStripLorentzAngleDQM_PSet.CondObj_fillId  = 'ProfileAndCumul'
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
    
    process.CondDataMonitoring.SiStripLowThresholdDQM_PSet.TkMap_On     = True
    process.CondDataMonitoring.SiStripLowThresholdDQM_PSet.TkMapName     = 'LowThresholdTkMap.png'
    process.CondDataMonitoring.SiStripLowThresholdDQM_PSet.minValue     = 0.
    process.CondDataMonitoring.SiStripLowThresholdDQM_PSet.maxValue     = 10.
    
    process.CondDataMonitoring.SiStripHighThresholdDQM_PSet.TkMap_On     = True
    process.CondDataMonitoring.SiStripHighThresholdDQM_PSet.TkMapName     = 'HighThresholdTkMap.png'
    process.CondDataMonitoring.SiStripHighThresholdDQM_PSet.minValue     = 0.
    process.CondDataMonitoring.SiStripHighThresholdDQM_PSet.maxValue     = 10.
    
    
    process.p1 = cms.Path(process.CondDataMonitoring)

# Additional analyzer if a bad channel tag has to be monitored
if options.QualityMon == True:
    process.SiStripQualityESProducer = cms.ESProducer("SiStripQualityESProducer",
                                                      ReduceGranularity = cms.bool(False),
                                                      PrintDebugOutput = cms.bool(False),
                                                      UseEmptyRunInfo = cms.bool(False),
                                                      ListOfRecordToMerge = cms.VPSet(cms.PSet(
        record = cms.string(options.recordName),
        tag = cms.string('')
        ))
                                                      )

# this module is almost useless since SiStripQualityDQM does all the job. If we want to remove it the log file has to be filled with SiStripQualityDQM
    process.stat = cms.EDAnalyzer("SiStripQualityStatistics",
                                  TkMapFileName = cms.untracked.string(''),
                                  dataLabel = cms.untracked.string('')
                                  )

    process.e = cms.EndPath(process.stat)

if options.CablingMon == True:
    process.SiStripQualityESProducer = cms.ESProducer("SiStripQualityESProducer",
                                                      ReduceGranularity = cms.bool(False),
                                                      PrintDebugOutput = cms.bool(False),
                                                      UseEmptyRunInfo = cms.bool(False),
                                                      ListOfRecordToMerge = cms.VPSet(cms.PSet(
        record = cms.string(options.recordForQualityName),
        tag = cms.string('')
        ))
                                                      )

    process.sistripconn = cms.ESProducer("SiStripConnectivity")

    process.stat = cms.EDAnalyzer("SiStripQualityStatistics",
                                  TkMapFileName = cms.untracked.string(''),
                                  dataLabel = cms.untracked.string('')
                                  )
    
    process.reader = cms.EDAnalyzer("SiStripFedCablingReader")
    
    process.e = cms.EndPath(process.stat*process.reader)
