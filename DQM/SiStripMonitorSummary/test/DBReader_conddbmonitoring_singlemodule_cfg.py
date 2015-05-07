import FWCore.ParameterSet.Config as cms

import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("Reader")

options = VarParsing.VarParsing("analysis")

options.register ('moduleList',
                  '',
                  VarParsing.VarParsing.multiplicity.list, # singleton or list
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "List of modules to monitor")
options.register ('gainNorm',
                  False,
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.bool,          # string, int, or float
                  "has gain normalization to be applied?")
options.register ('simGainNorm',
                  False,
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.bool,          # string, int, or float
                  "has SIM gain normalization to be applied?")
options.register ('globalTag',
                  "DONOTEXIST",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "GlobalTag")
options.register ('logDestination',
                  "",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "log file")
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

options.parseArguments()


process.MessageLogger = cms.Service("MessageLogger",
                                    debugModules = cms.untracked.vstring(''),
                                    destinations = cms.untracked.vstring(options.logDestination,
                                                                         'cerr'
                                                                         ), #Reader, cout
                                    categories = cms.untracked.vstring('SiStripBaseCondObjDQM',
                                                                       'SiStripNoisesDQM',
                                                                       'SiStripPedestalsDQM'
                                                                       ),
                                    cerr = cms.untracked.PSet(threshold = cms.untracked.string('WARNING'))
)
setattr(process.MessageLogger,options.logDestination,cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
        default = cms.untracked.PSet(limit=cms.untracked.int32(0)),
        SiStripBaseCondObjDQM = cms.untracked.PSet(limit=cms.untracked.int32(100000)),
        SiStripNoisesDQM = cms.untracked.PSet(limit=cms.untracked.int32(100000)),
        SiStripPedestalsDQM = cms.untracked.PSet(limit=cms.untracked.int32(100000))
))

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("EmptyIOVSource",
    firstValue = cms.uint64(options.runNumber),
    lastValue = cms.uint64(options.runNumber),
    timetype = cms.string('runnumber'),
    interval = cms.uint64(1)
)

if options.globalTag == "DONOTEXIST":
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
else:
    process.load("Configuration.StandardSequences.GeometryDB_cff")
    process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff")
    from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag 
    process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag, '')
    
process.DQMStore = cms.Service("DQMStore",
                               referenceFileName = cms.untracked.string(''),
                               verbose = cms.untracked.int32(1)
                               )

process.load("DQM.SiStripMonitorSummary.SiStripMonitorCondData_cfi")

process.CondDataMonitoring.OutputFileName = options.outputRootFile
process.CondDataMonitoring.MonitorSiStripPedestal      = options.PedestalMon
process.CondDataMonitoring.MonitorSiStripNoise         = options.NoiseMon
process.CondDataMonitoring.MonitorSiStripQuality       = False
process.CondDataMonitoring.MonitorSiStripCabling       = False
process.CondDataMonitoring.MonitorSiStripApvGain       = False
process.CondDataMonitoring.MonitorSiStripLorentzAngle  = False
process.CondDataMonitoring.MonitorSiStripBackPlaneCorrection  = False 
process.CondDataMonitoring.MonitorSiStripLowThreshold  = False
process.CondDataMonitoring.MonitorSiStripHighThreshold = False
process.CondDataMonitoring.OutputMEsInRootFile         = True
process.CondDataMonitoring.FillConditions_PSet.OutputSummaryAtLayerLevelAsImage           = False
process.CondDataMonitoring.FillConditions_PSet.OutputSummaryProfileAtLayerLevelAsImage    = False
process.CondDataMonitoring.FillConditions_PSet.OutputCumulativeSummaryAtLayerLevelAsImage = False
process.CondDataMonitoring.FillConditions_PSet.HistoMaps_On     = False
process.CondDataMonitoring.FillConditions_PSet.TkMap_On         = False
process.CondDataMonitoring.FillConditions_PSet.ActiveDetIds_On  = True
process.CondDataMonitoring.FillConditions_PSet.Mod_On           = True # Set to True if you want to have single module histograms
process.CondDataMonitoring.FillConditions_PSet.restrictModules  = True 
process.CondDataMonitoring.FillConditions_PSet.ModulesToBeIncluded = cms.vuint32(options.moduleList)

    
process.CondDataMonitoring.SiStripPedestalsDQM_PSet.FillSummaryAtLayerLevel     = True
process.CondDataMonitoring.SiStripNoisesDQM_PSet.FillSummaryAtLayerLevel        = True
    
process.CondDataMonitoring.SiStripPedestalsDQM_PSet.CondObj_fillId     = 'ProfileAndCumul'    # Use 'ProfileAndCumul' if you want to have single module histograms
process.CondDataMonitoring.SiStripNoisesDQM_PSet.CondObj_fillId        = 'ProfileAndCumul'      # Use 'ProfileAndCumul' if you want to have single module histograms
    
    
process.CondDataMonitoring.SiStripPedestalsDQM_PSet.TkMap_On     = False
process.CondDataMonitoring.SiStripPedestalsDQM_PSet.TkMapName    = 'PedestalTkMap.png'

process.CondDataMonitoring.SiStripNoisesDQM_PSet.TkMap_On     = False
process.CondDataMonitoring.SiStripNoisesDQM_PSet.TkMapName    = 'NoiseTkMap.png'
process.CondDataMonitoring.SiStripNoisesDQM_PSet.Cumul_NchX        = cms.int32(150)
process.CondDataMonitoring.SiStripNoisesDQM_PSet.Cumul_LowX        = cms.double(0.0)
process.CondDataMonitoring.SiStripNoisesDQM_PSet.Cumul_HighX       = cms.double(15.0)
process.CondDataMonitoring.SiStripNoisesDQM_PSet.GainRenormalisation               = cms.bool(options.gainNorm)
process.CondDataMonitoring.SiStripNoisesDQM_PSet.SimGainRenormalisation               = cms.bool(options.simGainNorm)

process.p1 = cms.Path(process.CondDataMonitoring)
