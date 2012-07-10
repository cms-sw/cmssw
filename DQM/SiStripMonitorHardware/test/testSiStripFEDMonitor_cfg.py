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
    #'/store/caf/user/venturia/nostripevents_Run2011A_prompt_jet_v4_160404-166502_v14.root',
    #'/store/data/Run2011A/MinimumBias/RAW/v1/000/167/913/14FFCB99-85A1-E011-B5DE-001D09F24259.root',
    '/store/data/Run2011A/MinimumBias/RAW/v1/000/166/683/062DC70A-1E91-E011-8998-0030487CD906.root',
    #'/store/data/Run2011A/MinimumBias/RAW/v1/000/166/683/0E2C8788-1A91-E011-A2F6-0030487CAEAC.root',
    #'/store/data/Run2011A/MinimumBias/RAW/v1/000/166/683/10A608E7-2791-E011-A849-0030487CD6E6.root',
    #'/store/data/Run2011A/MinimumBias/RAW/v1/000/166/683/3A606350-1691-E011-8E82-0030487C5CFA.root',
    #'/store/data/Run2011A/MinimumBias/RAW/v1/000/166/683/3C50CD4E-1B91-E011-8A01-0030487CD6E8.root',
    #'/store/data/Run2011A/MinimumBias/RAW/v1/000/166/683/4641277D-1F91-E011-AFE1-0030487CD6D2.root',
    #'/store/data/Run2011A/MinimumBias/RAW/v1/000/166/683/562E7E04-1791-E011-A319-0030487C7828.root',
    #'/store/data/Run2011A/MinimumBias/RAW/v1/000/166/683/66CB679F-1C91-E011-BC31-0030487CD6D8.root',
    #'/store/data/Run2011A/MinimumBias/RAW/v1/000/166/683/7C2122E0-1991-E011-A89D-003048F1110E.root',
    #'/store/data/Run2011A/MinimumBias/RAW/v1/000/166/683/88835770-1891-E011-9C0F-0030487CD718.root',
    #'/store/data/Run2011A/MinimumBias/RAW/v1/000/166/683/A05BEE57-1D91-E011-8435-0030487CD17C.root',
    #'/store/data/Run2011A/MinimumBias/RAW/v1/000/166/683/CE6060B3-1791-E011-AEC0-0030487CAF5E.root',
    ] )

process.source = cms.Source ("PoolSource",
                             fileNames=myfilelist
                             )

#process.load("DQM.SiStripMonitorHardware.test.source_cff")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
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
