import FWCore.ParameterSet.Config as cms

process = cms.Process('DQMFEDMonitor')

process.source = cms.Source(
  "PoolSource",
  fileNames = cms.untracked.vstring(
        #'file:/home/magnan/SOFTWARE/CMS/data/FED/edmOutput_run13.root'
        #'file:/home/magnan/SOFTWARE/CMS/data/FED/edmOutput_run85269.root'
        #'file:/home/magnan/SOFTWARE/CMS/data/FED/edmOutput_run96164.root'
        #'file:/home/magnan/SOFTWARE/CMS/data/FED/SiStripCommissioningSource_00096164_137.138.192.137_16715.root'
        #'file:/home/magnan/SOFTWARE/CMS/data/FED/Commissioning08/FEED31F3-58AC-DD11-BF73-000423D99658.root'
        #'file:/home/magnan/SOFTWARE/CMS/data/FED/Commissioning08/Run69750_FEED31F3-58AC-DD11-BF73-000423D99658.root'
        #'file:/home/magnan/SOFTWARE/CMS/data/FED/Commissioning08/Run69800_026DBE87-A5AC-DD11-9397-0030487C608C.root'
        #'file:/home/magnan/SOFTWARE/CMS/CMSSW_3_1_0_pre11/src/FedWorkDir/FedMonitoring/test/Digi_run69800.root'
        #'file:/home/magnan/SOFTWARE/CMS/data/FED/Commissioning08/Run69797_FC26431D-91AC-DD11-A0D1-001617E30CC8.root'
        #'file:/home/magnan/SOFTWARE/CMS/data/FED/Commissioning08/Run69874_98BB9120-E6AC-DD11-9B91-000423D99896.root'
        'file:/home/magnan/SOFTWARE/CMS/data/FED/Commissioning09/Run106019_00D9F347-4D72-DE11-93F6-001D09F24399.root'
        #'file:/home/magnan/SOFTWARE/CMS/data/FED/Commissioning09/Run101045_A6F7D0D3-4560-DE11-A52A-001D09F2545B.root'
 
        #'file:/home/magnan/SOFTWARE/CMS/data/FED/edmOutput_run82983.root'
        #'file:/home/magnan/SOFTWARE/CMS/data/FED/edmOutput_run77848.root'
        #'file:/home/magnan/SOFTWARE/CMS/data/FED/edmOutput_run79635.root'
        )
  )

#process.load("DQM.SiStripMonitorHardware.test.source_cff")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
    )

#process.service = cms.ProfilerService {
#    untracked int32 firstEvent = 1
#    untracked int32 lastEvent = 50
#    untracked vstring paths = { "p"}
#    }

#process.load('DQM.SiStripCommon.MessageLogger_cfi')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.MessageLogger = cms.Service(
    "MessageLogger",
    info = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
        #limit = cms.untracked.int32(100000),
        noLineBreaks = cms.untracked.bool(False)
        ),
    suppressInfo = cms.untracked.vstring(),
    # allows to suppress output from specific modules 
    suppressDebug = cms.untracked.vstring(),
    debug = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'),
        #limit = cms.untracked.int32(100000),
        noLineBreaks = cms.untracked.bool(False)
        ),
    warning = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING'),
        #limit = cms.untracked.int32(100000),
        noLineBreaks = cms.untracked.bool(False)
        ),
    cerr = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR'),
        #limit = cms.untracked.int32(100000),
        noLineBreaks = cms.untracked.bool(False)
        ),
    error = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR'),
        #limit = cms.untracked.int32(100000),
        noLineBreaks = cms.untracked.bool(False)
        ),
    suppressWarning = cms.untracked.vstring(),
    #debugModules = cms.untracked.vstring('*'),#'siStripFEDMonitor'),
    destinations = cms.untracked.vstring('cerr', 
                                         'debug', 
                                         'info', 
                                         'warning', 
                                         'error')

    )

process.DQMStore = cms.Service("DQMStore")

#needed to produce tkHistoMap
process.TkDetMap = cms.Service("TkDetMap")
process.SiStripDetInfoFileReader = cms.Service("SiStripDetInfoFileReader")

# Conditions (Global Tag is used here):
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.connect = "frontier://FrontierProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = "GR09_31X_V1P::All"
process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')


process.load('DQM.SiStripMonitorHardware.siStripFEDMonitor_cfi')
#process.load('DQM.SiStripMonitorHardware.siStripFEDMonitor_P5_cff')
#process.load('DQM.SiStripMonitorHardware.siStripFEDMonitor_Tier0_cff')
#process.siStripFEDMonitor.FillWithEventNumber = False
#process.siStripFEDMonitor.PrintDebugMessages = 2
process.siStripFEDMonitor.TimeHistogramConfig.NBins = 600
process.siStripFEDMonitor.TimeHistogramConfig.Min = 0
process.siStripFEDMonitor.TimeHistogramConfig.Max = 3600
process.siStripFEDMonitor.WriteDQMStore = True
process.siStripFEDMonitor.DQMStoreFileName = "DQMStore_monitoring_106019.root"
#process.siStripFEDMonitor.FillAllDetailedHistograms = False
process.Tracer = cms.Service('Tracer',indentation = cms.untracked.string('$$'))

process.load('PerfTools.Callgrind.callgrindSwitch_cff')

process.p = cms.Path( #process.profilerStart*
                      process.siStripFEDMonitor
                      #*process.profilerStop 
                      )
