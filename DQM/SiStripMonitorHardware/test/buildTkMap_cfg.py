import FWCore.ParameterSet.Config as cms

process = cms.Process('TKMAP')

process.maxEvents = cms.untracked.PSet(
   input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")


process.load('DQM.SiStripCommon.MessageLogger_cfi')
process.MessageLogger.debugModules = cms.untracked.vstring('*')
#process.MessageLogger.suppressInfo = cms.untracked.vstring('SiStripSpyDigiConverter')
#process.MessageLogger.suppressWarning = cms.untracked.vstring('SiStripSpyDigiConverter')
#process.MessageLogger.suppressDebug = cms.untracked.vstring('*')

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'GR10_P_V4::All'

process.DQMStore = cms.Service("DQMStore")

process.TkDetMap = cms.Service("TkDetMap")
process.SiStripDetInfoFileReader = cms.Service("SiStripDetInfoFileReader")


process.load('DQM.SiStripMonitorHardware.siStripBuildTrackerMap_cfi')
process.siStripBuildTrackerMap.InputFileName = '/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/data/OnlineData/00013xxxx/DQM_V0001_SiStrip_R000133016.root'
process.siStripBuildTrackerMap.InputFileNameForDiff = '/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/data/OnlineData/00013xxxx/DQM_V0001_SiStrip_R000133017.root'
process.siStripBuildTrackerMap.DoDifference = True
process.siStripBuildTrackerMap.MechanicalView = True
#process.siStripBuildTrackerMap.TkHistoMapNameVec = 'TkHMap_RunMeanGainPerCh','TkHMap_RunMeanZeroLightPerCh','TkHMap_RunMeanTickHeightPerCh','TkHMap_RunRmsGainPerCh','TkHMap_RunRmsZeroLightPerCh','TkHMap_RunRmsTickHeightPerCh'
process.siStripBuildTrackerMap.TkHistoMapNameVec = 'TkHMap_NumberOfOnTrackCluster','TkHMap_NumberOfOfffTrackCluster','TkHMap_NumberOfCluster','TkHMap_NumberOfDigi'
process.siStripBuildTrackerMap.MinValueVec = cms.untracked.vdouble(0,0,0,0)
process.siStripBuildTrackerMap.MaxValueVec = cms.untracked.vdouble(0,0,0,0)

process.siStripBuildTrackerMap.HistogramFolderName = ''
process.siStripBuildTrackerMap.PrintDebugMessages = 3

process.p = cms.Path(
    process.siStripBuildTrackerMap
    )

