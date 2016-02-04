import FWCore.ParameterSet.Config as cms

process = cms.Process('DQMFEDMonitor')

process.load('FWCore/MessageService/MessageLogger_cfi')
#process.MessageLogger.cerr.FwkReport.reportEvery = 1000

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

myfilelist = cms.untracked.vstring()

myfilelist.extend( [
    '/store/data/Run2011A/Cosmics/RAW/v1/000/161/847/001EC8F1-2F5C-E011-B85E-000423D9997E.root',
    '/store/data/Run2011A/Cosmics/RAW/v1/000/161/847/3A4CEFA4-295C-E011-B7E2-0030487CD14E.root',
    '/store/data/Run2011A/Cosmics/RAW/v1/000/161/847/3C5EC055-2A5C-E011-A607-0030487C6A66.root',
    '/store/data/Run2011A/Cosmics/RAW/v1/000/161/847/568F490D-2B5C-E011-8448-00304879BAB2.root',
    ] )

process.source = cms.Source ("PoolSource",
                             fileNames=myfilelist
                             )

#process.load("DQM.SiStripMonitorHardware.test.source_cff")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
    )

process.DQMStore = cms.Service("DQMStore")

#needed to produce tkHistoMap
process.TkDetMap = cms.Service("TkDetMap")
process.SiStripDetInfoFileReader = cms.Service("SiStripDetInfoFileReader")

# Conditions (Global Tag is used here):
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.connect = "frontier://FrontierProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = "GR_E_V14::All"
process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')


#process.load('DQM.SiStripMonitorHardware.siStripFEDMonitor_cfi')
process.load('DQM.SiStripMonitorHardware.siStripFEDMonitor_P5_cff')
#process.load('DQM.SiStripMonitorHardware.siStripFEDMonitor_Tier0_cff')
#process.siStripFEDMonitor.FillWithEventNumber = False
process.siStripFEDMonitor.PrintDebugMessages = 1
#process.siStripFEDMonitor.TimeHistogramConfig.NBins = 600
#process.siStripFEDMonitor.TimeHistogramConfig.Min = 0
#process.siStripFEDMonitor.TimeHistogramConfig.Max = 3600
process.siStripFEDMonitor.WriteDQMStore = True
process.siStripFEDMonitor.DQMStoreFileName = "DQMStore_FEDMonitoring.root"
#process.siStripFEDMonitor.FillAllDetailedHistograms = False
#process.Tracer = cms.Service('Tracer',indentation = cms.untracked.string('$$'))

process.load('PerfTools.Callgrind.callgrindSwitch_cff')

process.p = cms.Path( #process.profilerStart*
                      process.siStripFEDMonitor
                      #*process.profilerStop 
                      )
