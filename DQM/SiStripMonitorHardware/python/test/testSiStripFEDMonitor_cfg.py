import FWCore.ParameterSet.Config as cms

process = cms.Process('TEST')

process.source = cms.Source(
  "PoolSource",
  fileNames = cms.untracked.vstring(
        #'file:/home/magnan/SOFTWARE/CMS/data/FED/edmOutput_run13.root'
        #'file:/home/magnan/SOFTWARE/CMS/data/FED/edmOutput_run85269.root'
        #'file:/home/magnan/SOFTWARE/CMS/data/FED/edmOutput_run96164.root'
        #'file:/home/magnan/SOFTWARE/CMS/data/FED/SiStripCommissioningSource_00096164_137.138.192.137_16715.root'
        #'file:/home/magnan/SOFTWARE/CMS/data/FED/Commissioning08/FEED31F3-58AC-DD11-BF73-000423D99658.root'
        'file:/home/magnan/SOFTWARE/CMS/data/FED/Commissioning09/A6F7D0D3-4560-DE11-A52A-001D09F2545B.root'
        #'file:/home/magnan/SOFTWARE/CMS/data/FED/edmOutput_run82983.root'
        #'file:/home/magnan/SOFTWARE/CMS/data/FED/edmOutput_run77848.root'
        #'file:/home/magnan/SOFTWARE/CMS/data/FED/edmOutput_run79635.root'
        )
  )

#process.load("DQM.SiStripMonitorHardware.test.source_cff")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
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
#process.load('DQM.SiStripMonitorHardware.siStripFEDMonitor_Tier0_cff')
process.siStripFEDMonitor.FillWithEventNumber = True
process.siStripFEDMonitor.PrintDebugMessages = 2
process.siStripFEDMonitor.nTotalBadChannelsvsTimeHistogramConfig.Max = 1000;
process.siStripFEDMonitor.nTotalBadActiveChannelsvsTimeHistogramConfig.Max = 1000;
process.siStripFEDMonitor.nFEDErrorsvsTimeHistogramConfig.Max = 1000;
process.siStripFEDMonitor.nFEDCorruptBuffersvsTimeHistogramConfig.Max = 1000;
process.siStripFEDMonitor.nFEDsWithFEProblemsvsTimeHistogramConfig.Max = 1000;
process.siStripFEDMonitor.nAPVStatusBitvsTimeHistogramConfig.Max = 1000;
process.siStripFEDMonitor.nAPVErrorvsTimeHistogramConfig.Max = 1000;
process.siStripFEDMonitor.nAPVAddressErrorvsTimeHistogramConfig.Max = 1000;
process.siStripFEDMonitor.nUnlockedvsTimeHistogramConfig.Max = 1000;
process.siStripFEDMonitor.nOutOfSyncvsTimeHistogramConfig.Max = 1000;
process.siStripFEDMonitor.WriteDQMStore = True
process.siStripFEDMonitor.DQMStoreFileName = "DQMStore_comm09.root"
process.siStripFEDMonitor.FillAllDetailedHistograms = False
process.Tracer = cms.Service('Tracer',indentation = cms.untracked.string('$$'))

process.load('PerfTools.Callgrind.callgrindSwitch_cff')

process.p = cms.Path( #process.profilerStart*
                      process.siStripFEDMonitor
                      #*process.profilerStop 
                      )
