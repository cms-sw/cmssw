import FWCore.ParameterSet.Config as cms

process = cms.Process('BUILDTKMAP')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.load('FWCore/MessageService/MessageLogger_cfi')
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        noLineBreaks = cms.untracked.bool(False),
        threshold = cms.untracked.string('ERROR')
    ),
    files = cms.untracked.PSet(
        debug = cms.untracked.PSet(
            noLineBreaks = cms.untracked.bool(False),
            threshold = cms.untracked.string('DEBUG')
        ),
        error = cms.untracked.PSet(
            noLineBreaks = cms.untracked.bool(False),
            threshold = cms.untracked.string('ERROR')
        ),
        info = cms.untracked.PSet(
            noLineBreaks = cms.untracked.bool(False),
            threshold = cms.untracked.string('INFO')
        ),
        warning = cms.untracked.PSet(
            noLineBreaks = cms.untracked.bool(False),
            threshold = cms.untracked.string('WARNING')
        )
    ),
    suppressDebug = cms.untracked.vstring(),
    suppressInfo = cms.untracked.vstring(),
    suppressWarning = cms.untracked.vstring()
)


#process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "CRAFT0831X_V1::All"
#process.GlobalTag.globaltag = "GR09_31X_V1P::All"
process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')

#needed to produce tkHistoMap
process.load("DQM.SiStripCommon.TkHistoMap_cff")
# load TrackerTopology (needed for TkDetMap and TkHistoMap)
process.trackerTopology = cms.ESProducer("TrackerTopologyEP")

process.load('DQM.SiStripMonitorHardware.siStripBuildTrackerMap_cfi')
process.siStripBuildTrackerMap.InputFileName = '/home/magnan/SOFTWARE/CMS/data/FED/CMAnalysis/69797/CM_69797.root'
process.siStripBuildTrackerMap.MechanicalView = True
process.siStripBuildTrackerMap.TkHistoMapNameVec = 'TkHMap_MeanCMAPV0','TkHMap_MeanCMAPV1','TkHMap_MeanCMAPV0minusAPV1','TkHMap_RmsCMAPV0','TkHMap_RmsCMAPV1','TkHMap_RmsCMAPV0minusAPV1'
process.siStripBuildTrackerMap.HistogramFolderName = 'DQMData/'
process.siStripBuildTrackerMap.PrintDebugMessages = 2

process.p = cms.Path( process.siStripBuildTrackerMap
                      )
