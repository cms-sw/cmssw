import FWCore.ParameterSet.Config as cms

process = cms.Process('TKMAP')

process.maxEvents = cms.untracked.PSet(
   input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'GR10_P_V2::All'

process.DQMStore = cms.Service("DQMStore")

process.TkDetMap = cms.Service("TkDetMap")
process.SiStripDetInfoFileReader = cms.Service("SiStripDetInfoFileReader")


process.load('DQM.SiStripMonitorHardware.siStripBuildTrackerMap_cfi')
process.siStripBuildTrackerMap.InputFileName = 'DQMStore.root'
process.siStripBuildTrackerMap.MechanicalView = True
process.siStripBuildTrackerMap.TkHistoMapNameVec = 'TkHMap_RunMeanGainPerCh','TkHMap_RunMeanZeroLightPerCh','TkHMap_RunMeanTickHeightPerCh','TkHMap_RunRmsGainPerCh','TkHMap_RunRmsZeroLightPerCh','TkHMap_RunRmsTickHeightPerCh'
process.siStripBuildTrackerMap.HistogramFolderName = 'DQMData/'
process.siStripBuildTrackerMap.PrintDebugMessages = 3

process.p = cms.Path(
    process.siStripBuildTrackerMap
    )

