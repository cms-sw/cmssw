import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000
# Enable LogInfo
process.MessageLogger.cerr = cms.untracked.PSet(
    # threshold = cms.untracked.string('ERROR'),
    WARNING = cms.untracked.PSet(
        limit = cms.untracked.int32(0)
    ),
 )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.source = cms.Source("DQMRootSource",
                            fileNames = cms.untracked.vstring("file:OUT_step1.root")) # Files from step 1


process.load("HLTriggerOffline.Scouting.HLTScoutingEGammaPostProcessing_cff")
process.DQMStore = cms.Service("DQMStore")

#process.load("DQMServices.FileIO.DQMFileSaverOnline_cfi")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.dqmSaver.workflow = '/ScoutingElectron/myTest/DQM'
#process.dqmSaver.tag = 'SCOUTMONIT'
#process.dqmSaver.runNumber = 333334
#process.content = cms.EDAnalyzer("EventContentAnalyzer")

process.load("DQMServices.Components.MEtoEDMConverter_cff")
process.load("DQMServices.Components.DQMStoreStats_cfi")

#process.p = cms.Path(process.egmGsfElectronIDSequence + process.scoutingMonitoring + process.scoutingMonitoringTagProbe + process.scoutingMonitoringPatElectronTagProbe)
process.p = cms.Path(process.hltScoutingEGammaPostProcessing)
process.p1 = cms.Path(process.dqmSaver)
process.schedule = cms.Schedule(process.p, process.p1)
#process.p1 = cms.Path(process.scoutingEfficiencyHarvest)
