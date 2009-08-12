import FWCore.ParameterSet.Config as cms

process = cms.Process('BUILDTKMAP')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

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

process.load('DQM.SiStripMonitorHardware.siStripBuildTrackerMap_cfi')
process.siStripBuildTrackerMap.InputFileName = '/home/magnan/SOFTWARE/CMS/data/FED/CMAnalysis/69797/CM_69797.root'
process.siStripBuildTrackerMap.TkHistoMapNameVec = 'TkHMap_MeanCMAPV0','TkHMap_MeanCMAPV1','TkHMap_MeanCMAPV0minusAPV1','TkHMap_RmsCMAPV0','TkHMap_RmsCMAPV1','TkHMap_RmsCMAPV0minusAPV1'
process.siStripBuildTrackerMap.HistogramFolderName = 'DQMData/'
process.siStripBuildTrackerMap.PrintDebugMessages = 1

process.p = cms.Path( process.siStripBuildTrackerMap
                      )
